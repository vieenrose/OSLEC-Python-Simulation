/*
 *  Alternative OSLEC implementation using Improved Proportionate NLMS
 *  tailored for sparse echo paths and resource-limited systems.
 *
 *  This file provides a drop-in replacement for echo.c while preserving
 *  the exported API expected by kernel-space users of OSLEC.
 *
 *  Copyright (C) 2025  Open Source Contributors
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License version 2 as
 *  published by the Free Software Foundation.
 */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/types.h>
#include <linux/math64.h>
#include <linux/string.h>
#include "oslec.h"

#define Q16_SHIFT		16
#define Q16_ONE		(1 << Q16_SHIFT)
#define Q16_FROM_FLOAT(x)	((s32)((x) * (1 << Q16_SHIFT) + 0.5))

#define DEFAULT_MU_Q16		Q16_FROM_FLOAT(0.6f)
#define DEFAULT_ALPHA_Q16	Q16_FROM_FLOAT(0.5f)
#define DEFAULT_EPSILON		128

struct oslec_state {
	int taps;
	int head;
	int adaption_mode;
	s32 mu_q16;
	s32 alpha_q16;
	s32 epsilon;
	s16 *history;
	s32 *weights;
	s32 *gains;

	/* simple DC blocking filter state */
	s32 tx_prev;
	s32 tx_prev2;
};

static inline s32 q16_mul(s32 a, s32 b)
{
	return (s32)((s64)a * b >> Q16_SHIFT);
}

static inline s32 saturate16(s32 x)
{
	if (x > 32767)
		return 32767;
	if (x < -32768)
		return -32768;
	return x;
}

struct oslec_state *oslec_create(int len, int adaption_mode)
{
	struct oslec_state *ec;

	if (len <= 0)
		return NULL;

	ec = kzalloc(sizeof(*ec), GFP_KERNEL);
	if (!ec)
		return NULL;

	ec->history = kcalloc(len, sizeof(*ec->history), GFP_KERNEL);
	if (!ec->history)
		goto err_hist;

	ec->weights = kcalloc(len, sizeof(*ec->weights), GFP_KERNEL);
	if (!ec->weights)
		goto err_w;

	ec->gains = kcalloc(len, sizeof(*ec->gains), GFP_KERNEL);
	if (!ec->gains)
		goto err_g;

	ec->taps = len;
	ec->head = 0;
	ec->adaption_mode = adaption_mode;
	ec->mu_q16 = DEFAULT_MU_Q16;
	ec->alpha_q16 = DEFAULT_ALPHA_Q16;
	ec->epsilon = DEFAULT_EPSILON;
	ec->tx_prev = 0;
	ec->tx_prev2 = 0;

	return ec;

err_g:
	kfree(ec->weights);
err_w:
	kfree(ec->history);
err_hist:
	kfree(ec);
	return NULL;
}
EXPORT_SYMBOL_GPL(oslec_create);

void oslec_free(struct oslec_state *ec)
{
	kfree(ec->history);
	kfree(ec->weights);
	kfree(ec->gains);
	kfree(ec);
}
EXPORT_SYMBOL_GPL(oslec_free);

void oslec_adaption_mode(struct oslec_state *ec, int adaption_mode)
{
	ec->adaption_mode = adaption_mode;
}
EXPORT_SYMBOL_GPL(oslec_adaption_mode);

void oslec_flush(struct oslec_state *ec)
{
	memset(ec->history, 0, ec->taps * sizeof(*ec->history));
	memset(ec->weights, 0, ec->taps * sizeof(*ec->weights));
	memset(ec->gains, 0, ec->taps * sizeof(*ec->gains));
	ec->head = 0;
	ec->tx_prev = 0;
	ec->tx_prev2 = 0;
}
EXPORT_SYMBOL_GPL(oslec_flush);

void oslec_snapshot(struct oslec_state *ec)
{
	/* Left intentionally blank; snapshots not required by IPNLMS core. */
}
EXPORT_SYMBOL_GPL(oslec_snapshot);

static inline s16 fetch_history(const struct oslec_state *ec, int idx)
{
	int pos = ec->head + idx;

	if (pos >= ec->taps)
		pos -= ec->taps;

	return ec->history[pos];
}

static s32 compute_output(struct oslec_state *ec)
{
	s64 acc = 0;
	int i;

	for (i = 0; i < ec->taps; i++) {
		s32 w = ec->weights[i];
		s32 x = fetch_history(ec, i);

		acc += (s64)w * x;
	}

	acc >>= Q16_SHIFT;
	return saturate16(acc);
}

static void update_gains(struct oslec_state *ec)
{
	s64 sum_abs = 0;
	s32 alpha = ec->alpha_q16;
	s32 one_plus_alpha = Q16_ONE + alpha;
	s32 one_minus_alpha = Q16_ONE - alpha;
	int i;

	for (i = 0; i < ec->taps; i++)
		sum_abs += abs(ec->weights[i]);

	if (sum_abs == 0)
		sum_abs = 1;

	for (i = 0; i < ec->taps; i++) {
		u64 base = div_u64(one_minus_alpha, 2 * ec->taps);
		u64 denom = 2 * (u64)sum_abs + ec->epsilon;
		u64 prop = div_u64((u64)one_plus_alpha * abs(ec->weights[i]),
				   denom ? denom : 1);
		ec->gains[i] = (s32)(base + prop);
	}
}

static void adapt_weights(struct oslec_state *ec, s32 error)
{
	s64 norm = ec->epsilon;
	s32 mu = ec->mu_q16;
	int i;

	for (i = 0; i < ec->taps; i++) {
		s32 g = ec->gains[i];
		s32 x = fetch_history(ec, i);
		s64 term = (s64)g * x * x;
		norm += term;
	}

	if (norm == 0)
		norm = 1;

	for (i = 0; i < ec->taps; i++) {
		s32 g = ec->gains[i];
		s32 x = fetch_history(ec, i);
		s64 num = (s64)mu * g;

		num = (num * error) >> Q16_SHIFT;
		num = num * x;
		num = div_s64(num, norm);
		ec->weights[i] += (s32)num;
	}
}

int16_t oslec_update(struct oslec_state *ec, int16_t tx, int16_t rx)
{
	s32 est;
	s32 err;

	ec->head--;
	if (ec->head < 0)
		ec->head = ec->taps - 1;

	ec->history[ec->head] = tx;

	est = compute_output(ec);
	err = (s32)rx - est;

	if (ec->adaption_mode & ECHO_CAN_USE_ADAPTION) {
		update_gains(ec);
		adapt_weights(ec, err << Q16_SHIFT);
	}

	return saturate16(err);
}
EXPORT_SYMBOL_GPL(oslec_update);

int16_t oslec_hpf_tx(struct oslec_state *ec, int16_t tx)
{
	s32 y;

	y = tx - ec->tx_prev;
	ec->tx_prev2 = ec->tx_prev;
	ec->tx_prev = tx;

	return saturate16(y);
}
EXPORT_SYMBOL_GPL(oslec_hpf_tx);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Open Source Community");
MODULE_DESCRIPTION("IPNLMS-based acoustic echo canceller");
MODULE_VERSION("0.4.0");
