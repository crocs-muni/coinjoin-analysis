import os
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from html import escape

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FuncFormatter, NullFormatter, LogLocator, FixedLocator
import matplotlib.dates as mdates
import mpl_toolkits.axisartist as AA

import numpy as np
from matplotlib.transforms import ScaledTranslation
from scipy.optimize import minimize
from scipy import stats

import pandas as pd

from cj_process.cj_consts import *
from cj_process.cj_structs import *

from cj_process import cj_analysis as als
from cj_process import cj_visualize as cjvis

# SLOT_WIDTH_SECONDS = 3600 * 24 * 7  # week
#SLOT_WIDTH_SECONDS = 3600 * 24  # day
SLOT_WIDTH_SECONDS = 3600   # hour

LEGEND_FONT_SIZE = 8
#LEGEND_FONT_SIZE = 'small'
#LEGEND_FONT_SIZE = 'medium'

DEFAULT_AXIS_LABEL_SIZE = 14

def list_get(lst, idx, default=None):
    return lst[idx] if -len(lst) <= idx < len(lst) else default


def visualize_coinjoins_in_time(data, ax_num_coinjoins):
    #
    # Number of coinjoins per given time interval (e.g., day)
    #
    coinjoins = data["coinjoins"]
    broadcast_times = [precomp_datetime.strptime(coinjoins[item]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f") for item in
                       coinjoins.keys()]
    experiment_start_time = min(broadcast_times)
    slot_start_time = experiment_start_time
    slot_last_time = max(broadcast_times)
    diff_seconds = (slot_last_time - slot_start_time).total_seconds()
    num_slots = int(diff_seconds // SLOT_WIDTH_SECONDS)
    cjtx_in_hours = {hour: [] for hour in range(0, num_slots + 1)}
    rounds_started_in_hours = {hour: [] for hour in range(0, num_slots + 1)}
    for cjtx in coinjoins.keys():  # go over all coinjoin transactions
        timestamp = precomp_datetime.strptime(coinjoins[cjtx]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")
        cjtx_hour = int((timestamp - slot_start_time).total_seconds() // SLOT_WIDTH_SECONDS)
        cjtx_in_hours[cjtx_hour].append(cjtx)
    # remove last slot(s) if no coinjoins are available there
    while cjtx_in_hours[len(cjtx_in_hours.keys()) - 1] == []:
        del cjtx_in_hours[len(cjtx_in_hours.keys()) - 1]
    ax_num_coinjoins.plot([len(cjtx_in_hours[cjtx_hour]) for cjtx_hour in cjtx_in_hours.keys()],
                      label='All coinjoins finished', color='green')
    ax_num_coinjoins.legend()
    x_ticks = []
    for slot in cjtx_in_hours.keys():
        if SLOT_WIDTH_SECONDS < 3600:
            time_delta_format = "%Y-%m-%d %H:%M:%S"
        elif SLOT_WIDTH_SECONDS < 3600 * 24:
            time_delta_format = "%Y-%m-%d %H:%M:%S"
        else:
            time_delta_format = "%Y-%m-%d"
        x_ticks.append(
            (experiment_start_time + slot * timedelta(seconds=SLOT_WIDTH_SECONDS)).strftime(time_delta_format))
    ax_num_coinjoins.set_xticks(range(0, len(x_ticks)), x_ticks, rotation=45, fontsize=6)
    num_xticks = 30
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=num_xticks))
    ax_num_coinjoins.set_ylim(0)
    ax_num_coinjoins.set_ylabel('Number of coinjoin transactions')
    ax_num_coinjoins.set_title('Number of coinjoin transactions in given time period')



def visualize_coinjoins(mix_id, data, events, base_path, experiment_name):
    fig = plt.figure(figsize=(30, 20))
    ax_num_coinjoins = fig.add_subplot(4, 3, 1)
    ax_inputs_outputs = fig.add_subplot(4, 3, 2)
    ax_liquidity = fig.add_subplot(4, 3, 3)
    ax_input_time_to_burn = fig.add_subplot(4, 3, 4)
    ax_inputs_outputs_boxplot = fig.add_subplot(4, 3, 5)
    ax_liquidity_boxplot = fig.add_subplot(4, 3, 6)
    ax_inputs_value_bar = fig.add_subplot(4, 3, 7)
    ax_inputs_value_boxplot = fig.add_subplot(4, 3, 8)
    ax_fresh_inputs_value_boxplot = fig.add_subplot(4, 3, 9)
    ax_outputs_value_bar = fig.add_subplot(4, 3, 10)
    ax_outputs_value_boxplot = fig.add_subplot(4, 3, 11)
    ax_fresh_outputs_value_boxplot = fig.add_subplot(4, 3, 12)

    # Coinjoins in time
    visualize_coinjoins_in_time(data, ax_num_coinjoins)

    # All inputs and outputs
    visualize_liquidity_in_time(base_path, mix_id, data["coinjoins"], ax_inputs_outputs, ax_inputs_outputs_boxplot, ax_inputs_value_boxplot,
                                ax_outputs_value_boxplot, None, None, ax_input_time_to_burn, ['all inputs', 'all outputs', 'remixed inputs'])
    ax_inputs_outputs.set_ylabel('Number of inputs / outputs')
    ax_inputs_outputs.set_title('Number of all inputs and outputs in cjtx')
    ax_inputs_outputs_boxplot.set_ylabel('Number of inputs / outputs')
    ax_inputs_outputs_boxplot.set_title('Distribution of inputs of single coinjoins')

    ax_inputs_value_boxplot.set_ylabel('Value of inputs (sats)')
    ax_inputs_value_boxplot.set_yscale('log')
    ax_inputs_value_boxplot.set_title('Distribution of value of inputs (log scale)')
    ax_outputs_value_boxplot.set_ylabel('Value of outputs (sats)')
    ax_outputs_value_boxplot.set_yscale('log')
    ax_outputs_value_boxplot.set_title('Distribution of value of outputs (log scale)')

    ax_input_time_to_burn.set_ylabel('Input coin burn time (hours)')
    ax_input_time_to_burn.set_yscale('log')
    ax_input_time_to_burn.set_title('Distribution of coin burn times (log scale)')

    # Fresh liquidity in/out of mix
    visualize_liquidity_in_time(base_path, mix_id, events, ax_liquidity, ax_liquidity_boxplot, ax_fresh_inputs_value_boxplot,
                                ax_fresh_outputs_value_boxplot, ax_inputs_value_bar, ax_outputs_value_bar,
                                None, ['fresh inputs mixed', 'outputs leaving mix', ''], data.get('premix', None))
    ax_liquidity.set_ylabel('Number of new inputs / outputs')
    ax_liquidity.set_title('Number of fresh liquidity in and out of cjtx')
    ax_liquidity_boxplot.set_ylabel('Number of new inputs / outputs')
    ax_liquidity_boxplot.set_title('Distribution of fresh liquidity inputs of single coinjoins')
    ax_fresh_inputs_value_boxplot.set_ylabel('Value of inputs (sats)')
    ax_fresh_inputs_value_boxplot.set_yscale('log')
    ax_fresh_inputs_value_boxplot.set_title('Distribution of value of fresh inputs (log scale)')
    ax_fresh_outputs_value_boxplot.set_ylabel('Value of outputs (sats)')
    ax_fresh_outputs_value_boxplot.set_yscale('log')
    ax_fresh_outputs_value_boxplot.set_title('Distribution of value of fresh outputs (log scale)')

    ax_inputs_value_bar.set_ylabel('Number of inputs')
    ax_inputs_value_bar.set_title('Histogram of frequencies of specific values of fresh inputs.\n(x is log scale)')
    ax_outputs_value_bar.set_ylabel('Number of outputs')
    ax_outputs_value_bar.set_title('Histogram of frequencies of specific values of fresh outputs.\n(x is log scale)')

    # TODO: Add distribution of time-to-burn for remixed utxos
    # TODO: Add detection of any non-standard output values for WW2 and WW1

    # save graph
    plt.suptitle('{}'.format(experiment_name), fontsize=16)  # Adjust the fontsize and y position as needed
    plt.subplots_adjust(bottom=0.1, wspace=0.5, hspace=0.5)
    save_file = os.path.join(base_path, f'{experiment_name}_coinjoin_stats')
    plt.savefig(f'{save_file}.png', dpi=300)
    plt.savefig(f'{save_file}.pdf', dpi=300)
    plt.close()
    logging.info('Basic coinjoins statistics saved into {}'.format(save_file))


def visualize_liquidity_in_time(base_path, mix_id, events, ax_number, ax_boxplot, ax_input_values_boxplot, ax_output_values_boxplot,
                                ax_input_values_bar, ax_output_values_bar, ax_burn_time, legend_labels: list, events_premix: dict = None):
    #
    # Number of coinjoins per given time interval (e.g., day)
    #
    coinjoins = events
    broadcast_times_cjtxs = {item: precomp_datetime.strptime(coinjoins[item]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f") for item in
                       coinjoins.keys()}
    broadcast_times = list(broadcast_times_cjtxs.values())
    experiment_start_time = min(broadcast_times)
    slot_start_time = experiment_start_time
    slot_last_time = max(broadcast_times)
    diff_seconds = (slot_last_time - slot_start_time).total_seconds()
    num_slots = int(diff_seconds // SLOT_WIDTH_SECONDS)
    inputs_cjtx_in_slot = {hour: [] for hour in range(0, num_slots + 1)}
    outputs_cjtx_in_slot = {hour: [] for hour in range(0, num_slots + 1)}
    inputs_remixed_cjtx_in_slot = {hour: [] for hour in range(0, num_slots + 1)}
    inputs_values_in_slot = {hour: [] for hour in range(0, num_slots + 1)}
    tx0_inputs_values_in_slot = None
    outputs_values_in_slot = {hour: [] for hour in range(0, num_slots + 1)}
    inputs_burned_time_in_slot = {hour: [] for hour in range(0, num_slots + 1)}
    for cjtx in coinjoins.keys():  # go over all coinjoin transactions
        timestamp = precomp_datetime.strptime(coinjoins[cjtx]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")
        cjtx_hour = int((timestamp - slot_start_time).total_seconds() // SLOT_WIDTH_SECONDS)
        inputs_cjtx_in_slot[cjtx_hour].append(len(coinjoins[cjtx]['inputs']))
        outputs_cjtx_in_slot[cjtx_hour].append(len(coinjoins[cjtx]['outputs']))
        inputs_remixed_cjtx_in_slot[cjtx_hour].append(len([coinjoins[cjtx]['inputs'][index] for index in coinjoins[cjtx]['inputs'].keys()
                                                           if coinjoins[cjtx]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX.name]))
        inputs_values_in_slot[cjtx_hour].extend([coinjoins[cjtx]['inputs'][index]['value'] for index in coinjoins[cjtx]['inputs'].keys()])
        outputs_values_in_slot[cjtx_hour].extend([coinjoins[cjtx]['outputs'][index]['value'] for index in coinjoins[cjtx]['outputs'].keys()])

        # Extract difference in time between output creation and destruction in this cjtx
        if ax_burn_time:
            destruct_time = timestamp
            for index in coinjoins[cjtx]['inputs'].keys():
                if 'spending_tx' in coinjoins[cjtx]['inputs'][index].keys():
                    txid, vout = als.extract_txid_from_inout_string(coinjoins[cjtx]['inputs'][index]['spending_tx'])
                    if txid in broadcast_times_cjtxs.keys():
                        create_time = broadcast_times_cjtxs[txid]
                        time_diff = destruct_time - create_time
                        hours_diff = time_diff.total_seconds() / 3600
                        inputs_burned_time_in_slot[cjtx_hour].append(hours_diff)

    # If provided, process also TX0 premix
    if events_premix and len(events_premix) > 0:
        tx0_broadcast_times_cjtxs = {item: precomp_datetime.strptime(events_premix[item]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f") for
                                 item in events_premix.keys()}
        broadcast_times = list(tx0_broadcast_times_cjtxs.values())
        experiment_start_time = min(broadcast_times)
        slot_start_time = experiment_start_time
        slot_last_time = max(broadcast_times)
        diff_seconds = (slot_last_time - slot_start_time).total_seconds()
        num_slots = int(diff_seconds // SLOT_WIDTH_SECONDS)
        tx0_inputs_values_in_slot = {hour: [] for hour in range(0, num_slots + 1)}

        for cjtx in events_premix.keys():
            timestamp = precomp_datetime.strptime(events_premix[cjtx]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")
            cjtx_hour = int((timestamp - slot_start_time).total_seconds() // SLOT_WIDTH_SECONDS)
            tx0_inputs_values_in_slot[cjtx_hour].extend(
                [events_premix[cjtx]['inputs'][index]['value'] for index in events_premix[cjtx]['inputs'].keys()])
        while tx0_inputs_values_in_slot[len(tx0_inputs_values_in_slot.keys()) - 1] == []:
            del tx0_inputs_values_in_slot[len(tx0_inputs_values_in_slot.keys()) - 1]

    # remove last slot(s) if no coinjoins are available there
    while inputs_cjtx_in_slot[len(inputs_cjtx_in_slot.keys()) - 1] == []:
        del inputs_cjtx_in_slot[len(inputs_cjtx_in_slot.keys()) - 1]
    while outputs_cjtx_in_slot[len(outputs_cjtx_in_slot.keys()) - 1] == []:
        del outputs_cjtx_in_slot[len(outputs_cjtx_in_slot.keys()) - 1]
    while inputs_values_in_slot[len(inputs_values_in_slot.keys()) - 1] == []:
        del inputs_values_in_slot[len(inputs_values_in_slot.keys()) - 1]
    while outputs_cjtx_in_slot[len(outputs_cjtx_in_slot.keys()) - 1] == []:
        del outputs_values_in_slot[len(outputs_values_in_slot.keys()) - 1]

    ax_number.plot(range(0, len(inputs_cjtx_in_slot)), [sum(inputs_cjtx_in_slot[cjtx_hour]) for cjtx_hour in inputs_cjtx_in_slot.keys()],
                   label=legend_labels[0], alpha=0.5)
    ax_number.plot(range(0, len(outputs_cjtx_in_slot)), [sum(outputs_cjtx_in_slot[cjtx_hour]) for cjtx_hour in outputs_cjtx_in_slot.keys()],
                   label=legend_labels[1], alpha=0.5)
    ax_number.plot(range(0, len(inputs_remixed_cjtx_in_slot)), [sum(inputs_remixed_cjtx_in_slot[cjtx_hour]) for cjtx_hour in inputs_remixed_cjtx_in_slot.keys()],
                   label=legend_labels[2], alpha=0.5, linestyle='-.')

    # Create a boxplot
    data = [series[1] for series in inputs_cjtx_in_slot.items()]
    ax_boxplot.boxplot(data)
    # data = [series[1] for series in outputs_cjtx_in_slot.items()]
    # ax_boxplot.boxplot(data)
    data = [series[1] for series in inputs_values_in_slot.items()]
    ax_input_values_boxplot.boxplot(data)
    data = [series[1] for series in outputs_values_in_slot.items()]
    ax_output_values_boxplot.boxplot(data)
    if ax_burn_time:
        data = [series[1] for series in inputs_burned_time_in_slot.items()]
        ax_burn_time.boxplot(data)
        ax_burn_time.set_yscale('log')

    # Plot distribution of input values (bar height corresponding to number of occurences)
    if ax_input_values_bar:
        # For whirlpool, use distribution of inputs to TX0 (which splits inputs to premix), otherwise inputs to coinjoins
        if tx0_inputs_values_in_slot and len(tx0_inputs_values_in_slot) > 0:
            input_data = tx0_inputs_values_in_slot
        else:
            input_data = inputs_values_in_slot
        als.save_json_to_file_pretty(os.path.join(base_path, f'{mix_id}_inputs.json'), input_data)
        flat_data = [item for index in input_data.keys() for item in input_data[index]]
        log_data = np.log(flat_data)
        hist, bins = np.histogram(log_data, bins=100)
        ax_input_values_bar.bar(bins[:-1], hist, width=np.diff(bins))
        xticks = np.linspace(min(log_data), max(log_data), num=10)
        ax_input_values_bar.set_xticks(xticks, np.round(np.exp(xticks), decimals=0), rotation=45, fontsize=6)
        ax_input_values_bar.set_xlim(0, max(log_data))

    # Plot distribution of output values (bar height corresponding to number of occurences)
    if ax_output_values_bar:
        flat_data = [item for index in outputs_values_in_slot.keys() for item in outputs_values_in_slot[index]]
        if len(flat_data) > 0:
            log_data = np.log(flat_data)
            hist, bins = np.histogram(log_data, bins=100)
            ax_output_values_bar.bar(bins[:-1], hist, width=np.diff(bins))
            xticks = np.linspace(min(log_data), max(log_data), num=10)
            ax_output_values_bar.set_xticks(xticks, np.round(np.exp(xticks), decimals=0), rotation=45, fontsize=6)
            ax_output_values_bar.set_xlim(0, max(log_data))

    # if ax_burn_time:
    #     flat_data = [item for index in inputs_burned_time_in_slot.keys() for item in inputs_burned_time_in_slot[index]]
    #     ax_burn_time.bar(flat_data)

    ax_number.legend()
    ax_boxplot.legend()
    x_ticks = []
    for slot in inputs_cjtx_in_slot.keys():
        if SLOT_WIDTH_SECONDS < 3600:
            time_delta_format = "%Y-%m-%d %H:%M:%S"
        elif SLOT_WIDTH_SECONDS < 3600 * 24:
            time_delta_format = "%Y-%m-%d %H:%M:%S"
        else:
            time_delta_format = "%Y-%m-%d"

        x_ticks.append(
            (experiment_start_time + slot * timedelta(seconds=SLOT_WIDTH_SECONDS)).strftime(time_delta_format))
    ax_number.set_xticks(range(0, len(x_ticks)), x_ticks, rotation=45, fontsize=6)
    ax_boxplot.set_xticks(range(0, len(x_ticks)), x_ticks, rotation=45, fontsize=6)
    ax_input_values_boxplot.set_xticks(range(0, len(x_ticks)), x_ticks, rotation=45, fontsize=6)
    ax_output_values_boxplot.set_xticks(range(0, len(x_ticks)), x_ticks, rotation=45, fontsize=6)
    if ax_burn_time:
        ax_burn_time.set_xticks(range(0, len(x_ticks)), x_ticks, rotation=45, fontsize=6)
    num_xticks = 30
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=num_xticks))
    ax_number.set_ylim(0)
    ax_boxplot.set_ylim(0)
    if ax_burn_time:
        ax_burn_time.set_ylim(0)


def burntime_histogram(mix_id: str, data: dict):
    cjtxs = data["coinjoins"]
    burn_times = [cjtxs[cjtx]['inputs'][index]['burn_time_cjtxs']
                  for cjtx in cjtxs.keys() for index in cjtxs[cjtx]['inputs']
                  if 'burn_time_cjtxs' in cjtxs[cjtx]['inputs'][index].keys()]

    NUM_BINS = 1000
    # Compute standard histogram
    plt.figure()
    plt.hist(burn_times, NUM_BINS)
    plt.title(f'{mix_id} Histogram of burn times for all inputs')
    plt.xlabel('Burn time (num of coinjoins executed in meantime)')
    plt.ylabel('Frequency')
    plt.show()

    # Compute histogram in log scale
    plt.figure()
    hist, bins = np.histogram(burn_times,
                              bins=np.logspace(np.log10(min(burn_times)), np.log10(max(burn_times)), NUM_BINS))
    plt.bar(bins[:-1], hist, width=np.diff(bins), edgecolor='black')
    xticks = np.linspace(min(burn_times), max(burn_times), num=10)
    plt.xticks(xticks, np.round(np.exp(xticks), decimals=0), rotation=45, fontsize=6)
    plt.xscale('log')
    plt.title(f'{mix_id} Frequency of different burn times for all inputs')
    plt.xlabel('Burn time (num of coinjoins executed in meantime)')
    plt.ylabel('Frequency')
    plt.show()


def plot_autocorrelation(autocorr: list):
    plt.plot(autocorr)
    plt.title('Autocorrelation')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.show()


def plot_whirlpool_inputs_distribution(mix_id: str, inputs: list):
    log_data = np.log(inputs)
    hist, bins = np.histogram(log_data, bins=100)
    plt.bar(bins[:-1], hist, width=np.diff(bins))
    xticks = np.linspace(min(log_data), max(log_data), num=10)
    plt.xscale('log')
    plt.xticks(xticks, np.round(np.exp(xticks), decimals=0), rotation=45, fontsize=6)
    plt.title(f'{mix_id} inputs histogram (x axis is log)')
    plt.xlabel(f'Size of input')
    plt.ylabel(f'Number of inputs')
    plt.show()


def plot_analyze_liquidity(mix_id: str, cjtxs):
    plt.figure()

    short_exp_name = 'remix'
    sorted_cj_time = als.sort_coinjoins(cjtxs, als.SORT_COINJOINS_BY_RELATIVE_ORDER)

    #
    # num_unmixed_utxos_per_cj = [len(cjtxs[cjtx['txid']]['analysis2']['unmixed_utxos_in_wallets']) for cjtx in sorted_cj_time]
    # num_mixed_data_per_cj = [len(cjtxs[cjtx['txid']]['analysis2']['mixed_utxos_in_wallets']) for cjtx in sorted_cj_time]
    # num_finished_data_per_cj = [len(cjtxs[cjtx['txid']]['analysis2']['finished_utxos_in_wallets']) for cjtx in sorted_cj_time]
    # if ax:
    #     bar_width = 0.1
    #     categories = range(0, len(outputs_data))
    #     ax.bar(categories, num_unmixed_utxos_per_cj, bar_width,
    #                                  label=f'unmixed {short_exp_name}', alpha=0.3, color='blue')
    #     ax.bar(categories, num_mixed_data_per_cj, bar_width, bottom=num_unmixed_utxos_per_cj,
    #                                label=f'mixed {short_exp_name}', color='orange', alpha=0.3)
    #     ax.bar(categories, num_finished_data_per_cj, bar_width,
    #                                bottom=np.array(num_unmixed_utxos_per_cj) + np.array(num_mixed_data_per_cj),
    #                                label=f'finished {short_exp_name}', color='green', alpha=0.3)
    #     ax.plot(categories, num_unmixed_utxos_per_cj, label=f'unmixed {short_exp_name}',
    #                                   linewidth=3, color='blue', linestyle='--', alpha=0.3)
    #     ax.plot(categories, num_mixed_data_per_cj, label=f'mixed {short_exp_name}',
    #                                  linewidth=3, color='orange', linestyle='-.', alpha=0.3)
    #     ax.plot(categories, num_finished_data_per_cj, label=f'finished {short_exp_name}',
    #                                  linewidth=3, color='green', linestyle=':', alpha=0.3)
    #
    #     ax.set_xlabel('Coinjoin in time')
    #     ax.set_ylabel('Number of txos')
    #     ax.legend(loc='lower left')
    #     ax.set_title(f'Number of txos available in wallets when given cjtx is starting (all transactions)\n{experiment_name}')
    return None


def plot_wallets_distribution(target_path: str | Path, mix_id: str, factor: float, wallets_distrib: dict):
    labels = list(wallets_distrib.keys())
    values = list(wallets_distrib.values())
    plt.figure(figsize=(10, 3))
    plt.bar(labels, values)
    plt.title(f'{mix_id}: distribution of number of wallets in coinjoins (est. by factor {factor})')
    plt.xlabel(f'Number of wallets')
    plt.ylabel(f'Number of occurences')
    save_file = os.path.join(target_path, mix_id, f'{mix_id}_wallets_distribution_factor{factor}')
    plt.subplots_adjust(bottom=0.17)
    plt.savefig(f'{save_file}.png', dpi=300)
    plt.savefig(f'{save_file}.pdf', dpi=300)
    plt.close()


def inputs_value_burntime_heatmap(mix_id: str, data: dict):
    cjtxs = data["coinjoins"]
    # Create logarithmic range for values and digitize it
    # (we have too many different values, compute log bins then assign each precise value its bin number)
    NUM_BINS = 40  # Number of total bins to scale x and y axis to (logarithmically)

    # Sample list of tuples containing x and y coordinates
    points = [(cjtxs[cjtx]['inputs'][index]['burn_time_cjtxs'], cjtxs[cjtx]['inputs'][index]['value'])
              for cjtx in cjtxs.keys() for index in cjtxs[cjtx]['inputs'].keys()
              if 'burn_time_cjtxs' in cjtxs[cjtx]['inputs'][index].keys()]

    # Extract x and y coordinates from points list
    x_coords, y_coords = zip(*points)

    # Compute logarithmic bins for each axis separate (value of input, burn time in cjtxs)
    bins_x = np.logspace(np.log10(min(x_coords)), np.log10(max(x_coords)), num=NUM_BINS)
    bins_y = np.logspace(np.log10(min(y_coords)), np.log10(max(y_coords)), num=NUM_BINS)

    # Assign original precise values into range of log bins
    # np.digitize will compute corresponding bin for given precise value (5000 sats will go into first bin => 1...)
    x_coords_digitized = np.digitize(x_coords, bins_x)
    y_coords_digitized = np.digitize(y_coords, bins_y)
    points_digitized = zip(x_coords_digitized, y_coords_digitized)

    # Determine the dimensions of the heatmap (shall be close to NUM_BINS)
    x_min, x_max = min(x_coords_digitized), max(x_coords_digitized)
    y_min, y_max = min(y_coords_digitized), max(y_coords_digitized)
    x_range = np.arange(x_min, x_max + 1)
    y_range = np.arange(y_min, y_max + 1)
    # Create a grid representing the heatmap (initially empty)
    heatmap = np.zeros((len(y_range), len(x_range)))
    # Fill the grid with counts of points based in digitized inputs (value, burn_time)
    for x, y in points_digitized:
        heatmap[y - y_min, x - x_min] += 1

    # Plot the heatmap (no approximation)
    plt.figure()
    plt.hist2d(x_coords_digitized, y_coords_digitized, bins=NUM_BINS, cmap='plasma')
    plt.colorbar()

    # Add ticks labels from original non-log range
    custom_xticks = np.linspace(min(x_coords_digitized), max(x_coords_digitized), 10)
    custom_xticklabels = [f'{int(round(bins_x[int(tick-1)], 0))}' for tick in custom_xticks]  # Customize labels as needed
    plt.gca().set_xticklabels(custom_xticklabels, rotation=45, fontsize=6)
    custom_yticks = np.linspace(min(y_coords_digitized), max(y_coords_digitized), 10)
    custom_yticklabels = [f'{int(round(bins_y[int(tick)], 0))}' for tick in custom_yticks if int(tick) < len(bins_y)]  # Customize labels as needed
    plt.gca().set_yticklabels(custom_yticklabels, rotation=45, fontsize=6)
    plt.title(f'{mix_id} Input value to burn time heatmap for remixed coinjoin inputs')
    plt.xlabel('Burn time (num coinjoins)')
    plt.ylabel('Value of inputs (sats)')
    plt.show()



def visualize_mining_fees(mix_id: str, data: dict):
    cjtxs = data["coinjoins"]
    sorted_cj_time = als.sort_coinjoins(cjtxs, als.SORT_COINJOINS_BY_RELATIVE_ORDER)

    cjtxs_mining_fee = []
    for index in sorted_cj_time:
        cjtx = index['txid']
        inputs_val = sum([cjtxs[cjtx]['inputs'][index]['value'] for index in cjtxs[cjtx]['inputs'].keys()])
        outputs_val = sum([cjtxs[cjtx]['outputs'][index]['value'] for index in cjtxs[cjtx]['outputs'].keys()])
        cjtxs_mining_fee.append(inputs_val - outputs_val)

    print(f'Total mining fee: {sum(cjtxs_mining_fee) / SATS_IN_BTC} btc ({sum(cjtxs_mining_fee)} sats)')

    plt.figure()
    plt.plot(cjtxs_mining_fee)
    plt.title(f'{mix_id} Mining fee spent on coinjoin transactions in time')
    plt.xlabel('Index of coinjoin in time')
    plt.ylabel('Mining fee (sats)')
    plt.show()

    plt.figure()
    plt.hist(cjtxs_mining_fee, 100)
    plt.title(f'{mix_id} Histogram of mining fees spent on coinjoin transactions')
    plt.xlabel('Mining fee (sats)')
    plt.ylabel('Frequency')
    plt.show()

    #FEE_THRESHOLD = 1500000 # For WW2, almost all are below 1500000
    threshold = np.percentile(cjtxs_mining_fee, 95)
    filter_below_threshold = [value for value in cjtxs_mining_fee if value < threshold]

    plt.figure()
    plt.hist(filter_below_threshold, 100)
    plt.title(f'{mix_id} Histogram of mining fees spent on coinjoin transactions (95 percentil)')
    plt.xlabel('Mining fee (sats)')
    plt.ylabel('Frequency')
    plt.show()

    return cjtxs_mining_fee


def plot_wasabi_coordinator_fees(mix_id: str, cjtxs_coordinator_fee: list):
    plt.figure()
    plt.plot(cjtxs_coordinator_fee)
    plt.title(f'{mix_id} Coordination fee spent on coinjoin transactions in time')
    plt.xlabel('Index of coinjoin in time')
    plt.ylabel('Coordinator fee (sats)')
    plt.show()


def plot_whirlpool_coordinator_fees(mix_id: str, cjtxs_coordinator_fees: dict):
    plt.figure()
    for pool in cjtxs_coordinator_fees.keys():
        plt.plot(cjtxs_coordinator_fees[pool], label=f'{pool}')
    plt.title(f'{mix_id} Coordination fee spent by TX0 transactions in time')
    plt.xlabel('Index of coinjoin in time')
    plt.ylabel('Coordinator fee (sats)')
    plt.show()


def plot_month_year_separators(new_month_indices, separators_to_plot, ax2):
    # Plot lines as separators corresponding to months
    for pos in new_month_indices:
        if pos[0] in separators_to_plot:
            PLOT_DAYS_MONTHS = False
            if pos[0] == 'day' or pos[0] == 'month' and PLOT_DAYS_MONTHS:
                ax2.axvline(x=pos[1], color='gray', linewidth=0.5, alpha=0.1, linestyle='--')
            if pos[0] == 'year':
                ax2.axvline(x=pos[1], color='gray', linewidth=1, alpha=0.4, linestyle='--')
    ax2.set_xticks([x[1] for x in new_month_indices])
    labels = []
    prev_year_offset = -10000
    for x in new_month_indices:
        if x[0] == 'year':
            if x[1] - prev_year_offset > 1000:
                labels.append(f'{x[2][0:4]}')
                prev_year_offset = x[1]
            else:
                labels.append('')
        else:
            labels.append('')
    ax2.set_xticklabels(labels, rotation=45, fontsize=6)



def wasabi_plot_remixes_worker(mix_id: str, mix_protocol: MIX_PROTOCOL, target_path: Path, tx_file: str, sort_coinjoins_relative_order: bool,
                               analyze_values: bool = True, normalize_values: bool = True,
                               restrict_to_out_size = None, restrict_to_in_size = None,
                               plot_multi_graphs: bool=False, plot_single_intervals: bool=False, plot_aggregate: bool=False,
                               filter_paths: list=None, initial_ranges_values: dict=None):
    logging.info(f"Starting next worker")

    result = {}

    als.SORT_COINJOINS_BY_RELATIVE_ORDER = sort_coinjoins_relative_order

    files = os.listdir(target_path) if os.path.exists(target_path) else print(
        f'Path {target_path} does not exist')
    only_dirs = [file for file in files if os.path.isdir(os.path.join(target_path, file))]
    files = only_dirs
    if filter_paths is None:  # If filtering list is not provided, then process all paths
        filter_paths = files

    # Load fee rates
    mining_fee_rates = als.load_json_from_file(os.path.join(target_path, 'fee_rates.json'))

    # Load false positives
    false_cjtxs = als.load_false_cjtxs(target_path)

    # Compute number of required month subgraphs
    num_months = sum([1 for dir_name in files
                      if os.path.isdir(os.path.join(target_path, dir_name)) and
                      os.path.exists(os.path.join(target_path, dir_name, f'{tx_file}'))])

    if plot_multi_graphs or plot_aggregate:
        NUM_COLUMNS = 3
        NUM_ADDITIONAL_GRAPHS = 1 + NUM_COLUMNS
        NUM_ROWS = int((num_months + NUM_ADDITIONAL_GRAPHS) / NUM_COLUMNS + 1)
        fig = plt.figure(figsize=(40, NUM_ROWS * 5))

    ax_index = 1
    changing_liquidity = [0]  # Cummulative liquidity in mix from the perspective of given coinjoin (can go up and down)
    stay_liquidity = [0]  # Absolute cummulative liquidity staying in the mix outputs (mixed, but untouched)
    mining_fee_rate = []  # Mining fee rate
    remix_liquidity = [0] # Liquidity that is remixed in time despite likely reaching target anonscore
    changing_liquidity_timecutoff = [0]
    stay_liquidity_timecutoff = [0]
    coord_fee_rate = []  # Coordinator fee payments
    input_types = {}
    num_wallets = []
    initial_cj_index = 0
    time_liquidity = {}  # If MIX_LEAVE is detected, out liquidity is put into dictionary for future display
    no_remix_all = {'inputs': [], 'outputs': [], 'both': []}
    avg_input_ratio = {'all': [], 'per_interval': {}}

    prev_year = files[0][0:4]
    #new_month_indices = [('placeholder', 0, files[0][0:7])]  # Start with the first index
    new_month_indices = []
    next_month_index = 0
    days_dict = defaultdict(dict)
    weeks_dict = defaultdict(dict)
    months_dict = defaultdict(dict)

    for dir_name in sorted(files):
        if dir_name not in filter_paths:  # Process only for selected paths
            continue

        result[dir_name] = {}
        target_base_path = os.path.join(target_path, dir_name)
        tx_json_file = os.path.join(target_base_path, f'{tx_file}')
        current_year = dir_name[0:4]
        if os.path.isdir(target_base_path) and os.path.exists(tx_json_file):
            data = als.load_coinjoins_from_file(target_base_path, false_cjtxs, True)

            # If required, filter only coinjoins with specific size (whirlpool pools)
            if restrict_to_out_size is not None:
                before_len = len(data["coinjoins"])
                data["coinjoins"] = {cjtx: item for cjtx, item in data["coinjoins"].items() if
                                     restrict_to_out_size[0] <= item['outputs']['0']['value'] <=
                                     restrict_to_out_size[1]}
                print(f'Length after / before filtering {len(data["coinjoins"])} / {before_len} ({restrict_to_out_size[0]/SATS_IN_BTC} - {restrict_to_out_size[1]/SATS_IN_BTC})')
                if len(data["coinjoins"]) == 0:
                    print(f'No coinjoins of specified value {restrict_to_out_size[0]/SATS_IN_BTC} - {restrict_to_out_size[1]/SATS_IN_BTC} found in given interval, skipping')
                    continue

            fig_single = None
            if plot_single_intervals:
                fig_single, ax_to_use = plt.subplots(figsize=(20, 10))  # Figure for single plot
                #fig_single, ax_to_use = plt.subplots(figsize=(10, 2.5))  # Figure for single plot
            elif plot_multi_graphs:
                ax_to_use = fig.add_subplot(NUM_ROWS, NUM_COLUMNS, ax_index, axes_class=AA.Axes)  # Get next subplot
                ax_index += 1
            else:
                ax_to_use = None

            ax = ax_to_use

            # Plot lines as separators corresponding to days
            dates = sorted([precomp_datetime.strptime(data["coinjoins"][cjtx]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f") for cjtx in data["coinjoins"].keys()])
            new_day_indices = [('day', 0)]  # Start with the first index
            for i in range(1, len(dates)):
                if dates[i].day != dates[i - 1].day:
                    new_day_indices.append(('day', i))
            #print(new_day_indices)
            if ax:
                for pos in new_day_indices:
                    ax.axvline(x=pos[1], color='gray', linewidth=1, alpha=0.2)

            # Store index of coinjoins within this month (to be printed later in cummulative graph)
            if current_year == prev_year:
                new_month_indices.append(('month', next_month_index, dir_name[0:7]))
            else:
                new_month_indices.append(('year', next_month_index, dir_name[0:7]))
            next_month_index += len(data["coinjoins"])  # Store index of start fo next month (right after last index of current month)
            # TODO: if len(data["coinjoins"]) == 0, then month tick is plotted over previous one and is not visible
            #    add some artificial space to improve visibility? Or add artificial empty coinjoin bar? Or use different
            #    plotting style which will visibly plot also empty months?

            # Detect transactions with no remixes on input/out or both
            no_remix = als.detect_no_inout_remix_txs(data["coinjoins"])
            for key in no_remix.keys():
                if key not in no_remix_all.keys():
                    no_remix_all[key] = []
                no_remix_all[key].extend(no_remix[key])

            # Plot bars corresponding to different input types
            #plot_ax = ax if plot_multi_graphs else None
            plot_ax = ax

            input_types_interval = plot_inputs_type_ratio(f'{mix_id} {dir_name}', data, initial_cj_index, plot_ax, analyze_values, normalize_values, restrict_to_in_size)
            for input_type in input_types_interval:
                if input_type not in input_types.keys():
                    input_types[input_type] = []
                input_types[input_type].extend(input_types_interval[input_type])
            # Add current total mix liquidity into the same graph
            ax2 = ax.twinx() if ax else None
            #plot_ax = ax2 if plot_multi_graphs else None
            plot_ax = ax2

            # Set initial value for given intervale based on previous intervals (if provided)
            start_changing_liquidity = changing_liquidity[-1]
            start_stay_liquidity = stay_liquidity[-1]
            start_remix_liquidity = remix_liquidity[-1]
            start_changing_liquidity_timecutoff = changing_liquidity_timecutoff[-1]
            start_stay_liquidity_timecutoff = stay_liquidity_timecutoff[-1]
            if initial_ranges_values:  # If provided, use this as start values for liquidity instead
                start_changing_liquidity = initial_ranges_values[dir_name]['start_changing_liquidity']
                start_stay_liquidity = initial_ranges_values[dir_name]['start_stay_liquidity']
                start_remix_liquidity = initial_ranges_values[dir_name]['start_remix_liquidity']
                start_changing_liquidity_timecutoff = initial_ranges_values[dir_name]['start_changing_liquidity_timecutoff']
                start_stay_liquidity_timecutoff = initial_ranges_values[dir_name]['start_stay_liquidity_timecutoff']
            # Save initial value for the given interval
            result[dir_name]['start_changing_liquidity'] = start_changing_liquidity
            result[dir_name]['start_stay_liquidity'] = start_stay_liquidity
            result[dir_name]['start_remix_liquidity'] = start_remix_liquidity
            result[dir_name]['start_changing_liquidity_timecutoff'] = start_changing_liquidity_timecutoff
            result[dir_name]['start_stay_liquidity_timecutoff'] = start_stay_liquidity_timecutoff

            changing_liquidity_interval, stay_liquidity_interval, remix_liquidity_interval, changing_liquidity_timecutoff_interval, stay_liquidity_timecutoff_interval = (
                plot_mix_liquidity(f'{mix_id} {dir_name}', data, (start_changing_liquidity, start_stay_liquidity, start_remix_liquidity, start_changing_liquidity_timecutoff, start_stay_liquidity_timecutoff), time_liquidity, initial_cj_index, plot_ax))
            # Extend whole array over all intervals
            changing_liquidity.extend(changing_liquidity_interval)
            stay_liquidity.extend(stay_liquidity_interval)
            remix_liquidity.extend(remix_liquidity_interval)
            changing_liquidity_timecutoff.extend(changing_liquidity_timecutoff_interval)
            stay_liquidity_timecutoff.extend(stay_liquidity_timecutoff_interval)
            # Save whole interval results for
            result[dir_name]['interval_changing_liquidity'] = changing_liquidity_interval
            result[dir_name]['interval_stay_liquidity'] = stay_liquidity_interval
            result[dir_name]['interval_remix_liquidity'] = remix_liquidity_interval
            result[dir_name]['interval_changing_liquidity_timecutoff'] = changing_liquidity_timecutoff_interval
            result[dir_name]['interval_stay_liquidity_timecutoff'] = stay_liquidity_timecutoff_interval

            # Add fee rate into the same graph
            PLOT_FEERATE = False
            if PLOT_FEERATE and ax:
                ax3 = ax.twinx()
                ax3.spines['right'].set_position(('outward', -30))  # Adjust position of the third axis
            else:
                ax3 = None
                ax3_single = None
            plot_mining_fee_rates(f'{mix_id} {dir_name}', data, mining_fee_rates, ax3_single)
            mining_fee_rate_interval = plot_mining_fee_rates(f'{mix_id} {dir_name}', data, mining_fee_rates, ax3)
            mining_fee_rate.extend(mining_fee_rate_interval)

            PLOT_NUM_WALLETS = True if plot_single_intervals else False
            if PLOT_NUM_WALLETS and ax:
                ax3 = ax.twinx()
                ax3.spines['right'].set_position(('outward', -28))  # Adjust position of the third axis
            else:
                ax3 = None
            num_wallets_interval = plot_num_wallets(f'{mix_id} {dir_name}', data, avg_input_ratio, ax3)
            num_wallets.extend(num_wallets_interval)

            initial_cj_index = initial_cj_index + len(data["coinjoins"])
            if ax:
                ax.set_title(f'Type of inputs for given cjtx ({"values" if analyze_values else "number"})\n{mix_id} {dir_name}')
            logging.info(f'{target_base_path} inputs analyzed')

            # Compute liquidity inflows (sum of days/weeks/months)
            days_dict_interval, weeks_dict_interval, months_dict_interval = als.split_coinjoins_per_interval(data["coinjoins"], mix_protocol)
            days_dict.update(days_dict_interval)
            weeks_dict.update(weeks_dict_interval)
            months_dict.update(months_dict_interval)

            # Extend the y-limits to ensure the vertical lines go beyond the plot edges
            if ax:
                y_range = ax.get_ylim()
                padding = 0.02 * (y_range[1] - y_range[0])
                ax.set_ylim(y_range[0] - padding, y_range[1] + padding)

            # Save single interval figure
            if plot_single_intervals:
                plt.rcParams.update({'font.size': DEFAULT_AXIS_LABEL_SIZE})

                restrict_size_string = "" if restrict_to_in_size is None else f'{round(restrict_to_in_size[1] / SATS_IN_BTC, 3)}btc'
                save_file = os.path.join(target_path, dir_name,
                         f'{mix_id}_input_types_{"values" if analyze_values else "nums"}_{"norm" if normalize_values else "notnorm"}{restrict_size_string}')
                fig_single.savefig(f'{save_file}.png', dpi=300)
                fig_single.savefig(f'{save_file}.pdf', dpi=300)
                logging.debug(f'Successfully saved figure {save_file}')
                del ax
                del fig_single

        prev_year = current_year

        def plot_allcjtxs_cummulative(ax, new_month_indices, changing_liquidity, changing_liquidity_timecutoff, stay_liquidity, remix_liquidity, mining_fee_rate, separators_to_plot: list):
            # Plot mining fee rate
            PLOT_FEERATE = False
            if PLOT_FEERATE:
                ax.plot(mining_fee_rate, color='gray', alpha=0.3, linewidth=1, linestyle=':', label='Mining fee (90th percentil)')
                ax.tick_params(axis='y', colors='gray', labelsize=6)
                ax.set_ylabel('Mining fee rate sats/vB (90th percentil)', color='gray', fontsize='6', labelpad=-2)

            def plot_bars_downscaled(values, downscalefactor, color, ax):
                downscaled_values = [sum(values[i:i + downscalefactor]) for i in range(0, len(values), downscalefactor)]
                downscaled_indices = range(0, len(values), downscalefactor)
                ax.bar(downscaled_indices, downscaled_values, color=color, width=downscalefactor, alpha=0.2, edgecolor='none')

            # Create artificial limits if not provided
            if restrict_to_in_size is None:
                limit_size = (0, 1000000000000)
                print(f'No limits for inputs value')
            else:
                limit_size = restrict_to_in_size
                print(f'Limits for inputs value is {limit_size[0]} - {limit_size[1]}')

            # Decide on resolution of liquidity display
            #interval_to_display = months_dict
            #interval_to_display = weeks_dict
            interval_to_display = days_dict

            # TODO: replace this function by usage of als.compute_interval_aggregated_values()
            def compute_aggregated_interval_liquidity(interval_to_display):
                liquidity = [0]
                for interval in sorted(interval_to_display.keys()):
                    records = interval_to_display[interval]
                    mix_enter_values = [records[cjtx]['inputs'][index]['value'] for cjtx in records.keys() for index in
                                        records[cjtx]['inputs'].keys()
                                        if records[cjtx]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_ENTER.name or
                                        records[cjtx]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX_FRIENDS_WW1.name and
                                        limit_size[0] <= records[cjtx]['inputs'][index]['value'] <= limit_size[1]]
                    liquidity.extend([sum(mix_enter_values) / SATS_IN_BTC] * len(records))
                    print(f"Interval {interval}: {sum(mix_enter_values)}sats, num_cjtxs={len(records)}")
                return liquidity

            new_liquidity = compute_aggregated_interval_liquidity(interval_to_display)
            assert len(new_liquidity) == len(changing_liquidity), f'Incorrect enter_liquidity length: expected: {len(changing_liquidity)}, got {len(new_liquidity)}'
            plot_bars_downscaled(new_liquidity, 1, 'gray', ax)
            ax.set_title(f'{mix_id}: Liquidity dynamics in time')
            #label = f'{'Fresh liquidity (btc)' if analyze_values else 'Number of inputs'} {'normalized' if normalize_values else ''}'
            label = f'Fresh liquidity (btc)'
            ax.set_ylabel(label, color='gray', fontsize='6')
            ax.tick_params(axis='y', colors='gray')

            restrict_size_string = "" if restrict_to_in_size is None else f'{round(restrict_to_in_size[1] / SATS_IN_BTC, 3)}btc'
            save_file = os.path.join(target_path,
                             f'{mix_id}_freshliquidity_{"values" if analyze_values else "nums"}_{"norm" if normalize_values else "notnorm"}{restrict_size_string}')
            als.save_json_to_file_pretty(f'{save_file}.json', {'months_liquidity': compute_aggregated_interval_liquidity(months_dict), 'weeks_liquidity': compute_aggregated_interval_liquidity(weeks_dict)})

            # Outflows
            # out_liquidity = [input_types[MIX_EVENT_TYPE.MIX_LEAVE.name][i] for i in range(len(input_types[MIX_EVENT_TYPE.MIX_LEAVE.name]))]
            # plot_bars_downscaled(out_liquidity, 1, 'red', ax)

            # Remix ratio
            if MIX_EVENT_TYPE.MIX_REMIX.name not in input_types.keys():
                assert False, f'Missing MIX_REMIX for {target_base_path}'
            remix_ratios_all = [input_types[MIX_EVENT_TYPE.MIX_REMIX.name][i] * 100 for i in
                                range(len(input_types[MIX_EVENT_TYPE.MIX_REMIX.name]))]  # All remix including nonstandard
            remix_ratios_nonstd = [input_types['MIX_REMIX_nonstd'][i] * 100 for i in
                                   range(len(input_types['MIX_REMIX_nonstd']))]  # Nonstd remixes
            remix_ratios_std = [remix_ratios_all[i] - remix_ratios_nonstd[i] for i in
                                range(len(remix_ratios_all))]  # Only standard remixes
            WINDOWS_SIZE = round(len(remix_ratios_all) / 1000)  # Set windows size to get 1000 points total (unless short, then only 5)
            WINDOWS_SIZE = 1 if WINDOWS_SIZE < 1 else WINDOWS_SIZE
            if mix_protocol == MIX_PROTOCOL.WASABI1:
                # Wasabi 1 ix only single output per denonimation, putting automatically (potentially large) change into next remix
                # Compute remix rate only from standard denomination inputs as large remix fraction are these change remixes which are
                # easily distinguishable from standard denomination inputs
                remix_ratios_avg = [np.average(remix_ratios_std[i:i + WINDOWS_SIZE]) for i in
                                    range(0, len(remix_ratios_std), WINDOWS_SIZE)]
            else:
                # Consider all inputs from non-wasabi1 pools
                remix_ratios_avg = [np.average(remix_ratios_all[i:i + WINDOWS_SIZE]) for i in
                                    range(0, len(remix_ratios_all), WINDOWS_SIZE)]

            ax2 = ax.twinx()
            ax2.plot(range(0, len(remix_ratios_std), WINDOWS_SIZE), remix_ratios_avg, label=f'MIX_REMIX avg({WINDOWS_SIZE})',
                     color='brown', linewidth=1, linestyle='--', alpha=0.5)
            ax2.set_ylim(0, 100)  # Force whole range of yaxis
            ax2.tick_params(axis='y', colors='brown', labelsize=6)
            ax2.set_ylabel('Average remix rate %', color='brown', fontsize='6', labelpad=-3)
            ax2.spines['right'].set_position(('outward', -25))  # Adjust position of the third axis

            # Save computed remixes to file
            restrict_size_string = "" if restrict_to_in_size is None else f'{round(restrict_to_in_size[1] / SATS_IN_BTC, 3)}btc'
            save_file = os.path.join(target_path,
                             f'{mix_id}_remixrate_{"values" if analyze_values else "nums"}_{"norm" if normalize_values else "notnorm"}{restrict_size_string}')
            als.save_json_to_file_pretty(f'{save_file}.json', {'remix_ratios_all': remix_ratios_all, 'remix_ratios_nonstd': remix_ratios_nonstd, 'remix_ratios_std': remix_ratios_std})

            # Plot changing liquidity in time
            ax2 = ax.twinx()
            changing_liquidity_btc = [item / SATS_IN_BTC for item in changing_liquidity]
            changing_liquidity_timecutoff_btc = [item / SATS_IN_BTC for item in changing_liquidity_timecutoff]
            remix_liquidity_btc = [item / SATS_IN_BTC for item in remix_liquidity]
            stay_liquidity_btc = [item / SATS_IN_BTC for item in stay_liquidity]
            ax2.plot(changing_liquidity_btc, color='royalblue', alpha=0.6, linewidth=2, label='Interim liquidity (MIX_ENTER - MIX_LEAVE)')
            ax2.plot(stay_liquidity_btc, color='darkgreen', alpha=0.6, linestyle='--', label='Unmoved outputs (MIX_STAY)')
            #ax2.plot(remix_liquidity_btc, color='black', alpha=0.6, linestyle='--', label='Cummulative remix liquidity, MIX_ENTER - MIX_LEAVE - MIX_STAY')
            ax2.plot([0], [0], label=f'Average remix rate', color='brown', linewidth=1, linestyle='--', alpha=0.5)  # Fake plot to have correct legend record from other twinx
            ax2.plot([0], [0], color='gray', alpha=0.2, linestyle='-', label='Fresh daily liquidity inflows')

            PLOT_CHAINANALYSIS_TIMECUTOFF = False
            if PLOT_CHAINANALYSIS_TIMECUTOFF:
                ax2.plot(changing_liquidity_timecutoff_btc, color='blue', alpha=0.6,
                         label='Interim liquidity (MIX_ENTER - MIX_LEAVE, time cutoff)')
                #ax2.plot([a - b for a, b in zip([item / SATS_IN_BTC for item in changing_liquidity_timecutoff], [item / SATS_IN_BTC for item in stay_liquidity_timecutoff])], color='blue', alpha=0.6, linestyle='-.', label='Actively remixed liquidity (Changing - Unmoved)')

            ax2.plot([a - b for a, b in zip(changing_liquidity_btc, stay_liquidity_btc)], color='red', alpha=0.6, linestyle='-.', label='Actively remixed liquidity (Interim - Unmoved)')
            ax2.set_ylabel('btc in mix', color='royalblue')
            ax2.tick_params(axis='y', colors='royalblue')

            ax3 = None
            PLOT_ESTIMATED_WALLETS = False
            if PLOT_ESTIMATED_WALLETS:
                # TODO: Compute wallets estimation based on inputs per time interval, not directly conjoins
                AVG_WINDOWS = 10
                num_wallets_avg = als.compute_averages(num_wallets, AVG_WINDOWS)
                AVG_WINDOWS_100 = 100
                num_wallets_avg100 = als.compute_averages(num_wallets, AVG_WINDOWS_100)
                ax3 = ax.twinx()
                ax3.spines['right'].set_position(('outward', -28))  # Adjust position of the third axis
                ax3.plot(num_wallets_avg, color='green', alpha=0.4, label=f'Estimated # wallets ({AVG_WINDOWS} avg)')
                ax3.plot(num_wallets_avg100, color='green', alpha=0.8, label=f'Estimated # wallets ({AVG_WINDOWS_100} avg)')
                ax3.set_ylabel('Estimated number of active wallets', color='green')
                ax3.tick_params(axis='y', colors='green')

            # Plot lines as separators corresponding to months
            plot_month_year_separators(new_month_indices, separators_to_plot, ax2)

            # if ax:
            #     ax.legend(loc='center left')
            if ax2:
                if mix_protocol in [MIX_PROTOCOL.WASABI2]:
                    ax2.legend(loc='upper left', fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.01, 0.85), borderaxespad=0)
                else:
                    ax2.legend(loc='upper left', fontsize=LEGEND_FONT_SIZE)
            if ax3:
                ax3.legend()

    # Save input_types into json for plotly external plotting
    PLOT_PLOTLY = False
    if PLOT_PLOTLY:
        plotly_data = {'time': list(range(0, len(input_types[MIX_EVENT_TYPE.MIX_REMIX.name])))}
        for input_type in input_types.keys():
            if input_type in [MIX_EVENT_TYPE.MIX_ENTER.name, MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name,
                              MIX_EVENT_TYPE.MIX_REMIX_FRIENDS_WW1.name, 'MIX_REMIX_1', 'MIX_REMIX_2', 'MIX_REMIX_3-5',
                              'MIX_REMIX_6-19', 'MIX_REMIX_20+', 'MIX_REMIX_1000-1999', 'MIX_REMIX_2000+',
                              'MIX_REMIX_nonstd']:
                plotly_data[input_type] = [value.item() for value in input_types[input_type]]
        save_file = os.path.join(target_path, 'plotly_data.json')
        als.save_json_to_file(save_file, plotly_data)

    if plot_multi_graphs:
        # Add additional cummulative plots for all coinjoin in one
        ax = fig.add_subplot(NUM_ROWS, NUM_COLUMNS, ax_index, axes_class=AA.Axes)  # Get next subplot
        ax_index += 1
        plot_allcjtxs_cummulative(ax, new_month_indices, changing_liquidity, changing_liquidity_timecutoff, stay_liquidity, remix_liquidity, mining_fee_rate, ['month', 'year'])

        # Finalize multigraph graph
        plt.subplots_adjust(bottom=0.1, wspace=0.25, hspace=0.4)
        plt.rcParams.update({'font.size': DEFAULT_AXIS_LABEL_SIZE})
        restrict_size_string = "" if restrict_to_in_size is None else f'{round(restrict_to_in_size[1] / SATS_IN_BTC, 3)}btc'
        save_file = os.path.join(target_path, f'{mix_id}_input_types_{"values" if analyze_values else "nums"}_{"norm" if normalize_values else "notnorm"}{restrict_size_string}')
        plt.savefig(f'{save_file}.png', dpi=300)
        plt.savefig(f'{save_file}.pdf', dpi=300)
        # with open(f'{save_file}.html', "w") as f:
        #     f.write(mpld3.fig_to_html(plt.gcf()))
        plt.close()

        # Save generate and save cummulative results separately
    if plot_aggregate:
        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(1, 1, 1, axes_class=AA.Axes)  # Get next subplot
        plot_allcjtxs_cummulative(ax, new_month_indices, changing_liquidity, changing_liquidity_timecutoff, stay_liquidity, remix_liquidity, mining_fee_rate, ['month', 'year'])
        plt.subplots_adjust(bottom=0.1, wspace=0.15, hspace=0.4)
        plt.rcParams.update({'font.size': DEFAULT_AXIS_LABEL_SIZE})
        restrict_size_string = "" if restrict_to_in_size is None else f'{round(restrict_to_in_size[1] / SATS_IN_BTC, 3)}btc'
        save_file = os.path.join(target_path, f'{mix_id}_cummul_{"values" if analyze_values else "nums"}_{"norm" if normalize_values else "notnorm"}{restrict_size_string}')
        plt.savefig(f'{save_file}.png', dpi=300)
        plt.savefig(f'{save_file}.pdf', dpi=300)
        save_file = os.path.join(target_path, f'{mix_id}_cummul_{"values" if analyze_values else "nums"}_{"norm" if normalize_values else "notnorm"}{restrict_size_string}_nolegend')
        plt.legend().set_visible(False)
        plt.savefig(f'{save_file}.png', dpi=300)
        plt.savefig(f'{save_file}.pdf', dpi=300)
        # with open(f'{save_file}.html', "w") as f:
        #     f.write(mpld3.fig_to_html(plt.gcf()))
        plt.close()

    # save detected transactions with no remixes (potentially false positives)
    als.save_json_to_file_pretty(os.path.join(target_path, 'no_remix_txs_simplified.json'), no_remix_all)

    # store also results over all intervals
    result['all_changing_liquidity'] = changing_liquidity
    result['all_stay_liquidity'] = stay_liquidity
    result['all_remix_liquidity'] = remix_liquidity
    result['all_changing_liquidity_timecutoff'] = changing_liquidity_timecutoff
    result['all_stay_liquidity_timecutoff'] = stay_liquidity_timecutoff
    return result


def run_estimate_wallet_prediction_factor(target_path: str, coord: str, conf_interval: str, plot_inputs_predictions: bool, plot_outputs_predictions: bool):
    if coord == 'wasabi2_zksnacks':
        predict_matrix = als.load_json_from_file(
            os.path.join(target_path, 'wallet_estimation_matrix_ww2zksnacks.json'))
    else:
        predict_matrix = als.load_json_from_file(
            os.path.join(target_path, 'wallet_estimation_matrix_ww2kruw.json'))
    all_data = als.load_coinjoins_from_file(os.path.join(target_path, coord), None, True)

    cjvis.estimate_wallet_prediction_factor(all_data, target_path, coord, predict_matrix[conf_interval],
                                            plot_inputs_predictions, plot_outputs_predictions)


def estimate_wallet_prediction_factor(all_data: dict, base_path, mix_id, prediction_matrix: dict=None,
                                      plot_inputs_prediction: bool=True, plot_outputs_prediction: bool=True, ax_provided=None, do_plot=True):
    # REFACTOR - mixed analysis and plotting
    AVG_NUM_INPUTS, AVG_NUM_OUTPUTS = als.get_wallets_prediction_ratios(mix_id, prediction_matrix)

    target_load_path = os.path.join(base_path, mix_id)
    sorted_cj_time = als.sort_coinjoins(all_data['coinjoins'], als.SORT_COINJOINS_BY_RELATIVE_ORDER)

    if len(sorted_cj_time) < 2:
        logging.warning(f'estimate_wallet_prediction_factor() - too little coinjoins available for {mix_id}, not continuing')
        return []

    logging.debug(f'estimate_wallet_prediction_factor() going to estimate input factors for {mix_id}')

    num_all_inputs = np.array([len(all_data['coinjoins'][cj['txid']]['inputs']) for cj in sorted_cj_time])
    num_all_outputs = np.array([len(all_data['coinjoins'][cj['txid']]['outputs']) for cj in sorted_cj_time])

    # Find heuristically the AVG_NUM_INPUTS and AVG_NUM_OUTPUTS to minimize difference between computed number of inputs and outputs
    # Objective function to minimize
    def objective_linear(params, x_window, y_window):
        x1, y1 = params
        return np.sum(np.abs(x_window / x1 - y_window / y1))

    # Store index of coinjoins when it changes months/years
    new_month_indices = []
    next_month_index = 0
    dates = sorted(
        [precomp_datetime.strptime(all_data["coinjoins"][cjtx['txid']]['broadcast_time_virtual'], "%Y-%m-%d %H:%M:%S.%f") for cjtx in
         sorted_cj_time])
    for i in range(1, len(dates)):
        if dates[i].month != dates[i - 1].month:
            if dates[i].year == dates[i - 1].year:
                new_month_indices.append(('month', next_month_index, f'{dates[i].year}-{dates[i].month:02}'))
            else:
                new_month_indices.append(('year', next_month_index, f'{dates[i].year}-{dates[i].month:02}'))
        next_month_index = next_month_index + 1

    ratios_list_every_cjtx = [num_all_inputs[offset] / (num_all_outputs[offset] / AVG_NUM_OUTPUTS) for offset in range(0, len(num_all_inputs))]  # Number of wallets in every

    ratios_list = []
    WINDOW_LEN = 10
    for offset in range(0, len(num_all_inputs) - WINDOW_LEN):
        # Initial guess for x1 and y1
        initial_guess = [1, 1]
        x_window = num_all_inputs[offset:offset + WINDOW_LEN]
        y_window = num_all_outputs[offset:offset + WINDOW_LEN]
        # Minimize the objective function
        result = minimize(objective_linear, initial_guess, args=(x_window, y_window), method='Nelder-Mead')
        # Optimal values
        x1_opt, y1_opt = result.x
        AVG_NUM_INPUTS = AVG_NUM_OUTPUTS * (x1_opt / y1_opt)
        ratios_list.append(AVG_NUM_INPUTS)

    #
    # Compute number of predicted wallets
    COLOR_WALLETS_INPUTS = 'red'
    COLOR_WALLETS_OUTPUTS = 'green'
    predicted_wallets_list_inputs = []
    predicted_wallets_list_inputs_cilo = []
    predicted_wallets_list_inputs_cihi = []
    predicted_wallets_list_outputs = []
    predicted_wallets_list_outputs_cilo = []
    predicted_wallets_list_outputs_cihi = []
    # Select ratios to use
    used_prediction_ratios = ratios_list

    LARGE_AVG_WINDOW = 100
    ratios_list_avg = als.smooth_interval(ratios_list, LARGE_AVG_WINDOW)

    last_usable_factor = used_prediction_ratios[0]
    if prediction_matrix:
        for i in range(0, len(sorted_cj_time)):
            num_inputs = len(all_data['coinjoins'][sorted_cj_time[i]['txid']]['inputs'].keys())
            predicted_wallets_list_inputs.append(prediction_matrix['inputs'][str(num_inputs)]['N_hat'])
            predicted_wallets_list_inputs_cilo.append(prediction_matrix['inputs'][str(num_inputs)]['ci_lo'])
            predicted_wallets_list_inputs_cihi.append(prediction_matrix['inputs'][str(num_inputs)]['ci_hi'])

            num_outputs = len(all_data['coinjoins'][sorted_cj_time[i]['txid']]['outputs'].keys())
            predicted_wallets_list_outputs.append(prediction_matrix['outputs'][str(num_outputs)]['N_hat'])
            predicted_wallets_list_outputs_cilo.append(prediction_matrix['outputs'][str(num_outputs)]['ci_lo'])
            predicted_wallets_list_outputs_cihi.append(prediction_matrix['outputs'][str(num_outputs)]['ci_hi'])
    else:
        for i in range(0, len(sorted_cj_time)):
            # Use computed prediction factor if available
            if list_get(used_prediction_ratios, i, -1) != -1:
                predicted_num_wallets = int(round(num_all_inputs[i] / used_prediction_ratios[i]))
                last_usable_factor = ratios_list[i]
            else:
                # Last last known if factor no longer computed (due to size of average window)
                predicted_num_wallets = int(round(num_all_inputs[i] / last_usable_factor))
            predicted_wallets_list_inputs.append(predicted_num_wallets)
            predicted_wallets_list_outputs.append(int(round(num_all_outputs[i] / AVG_NUM_OUTPUTS)))

    # Compute averages
    predicted_wallets_inputs_avg = als.smooth_interval(predicted_wallets_list_inputs, LARGE_AVG_WINDOW)
    predicted_wallets_inputs_cilo_avg = als.smooth_interval(predicted_wallets_list_inputs_cilo, LARGE_AVG_WINDOW) if predicted_wallets_list_inputs_cilo else None
    predicted_wallets_inputs_cihi_avg = als.smooth_interval(predicted_wallets_list_inputs_cihi, LARGE_AVG_WINDOW) if predicted_wallets_list_inputs_cihi else None
    predicted_wallets_outputs_avg = als.smooth_interval(predicted_wallets_list_outputs, LARGE_AVG_WINDOW)
    predicted_wallets_outputs_cilo_avg = als.smooth_interval(predicted_wallets_list_outputs_cilo, LARGE_AVG_WINDOW) if predicted_wallets_list_outputs_cilo else None
    predicted_wallets_outputs_cihi_avg = als.smooth_interval(predicted_wallets_list_outputs_cihi, LARGE_AVG_WINDOW) if predicted_wallets_list_outputs_cihi else None

    if ax_provided == None:
        fig_single, ax = plt.subplots(figsize=(12, 4))  # Figure for single plot
    else:
        ax = ax_provided
    # Plot explict time ticks instead of default ones
    plot_month_year_separators(new_month_indices, ['month', 'year'], ax)
    ax.set_xlabel('coinjoin in time')
    plt.xticks(fontsize=DEFAULT_AXIS_LABEL_SIZE)

    PLOT_NUM_WALLETS = True
    FULL_LEGEND = False
    if PLOT_NUM_WALLETS and do_plot:
        if plot_inputs_prediction:
            ax.plot(predicted_wallets_list_inputs,
                     label='Predicted # wallets (inputs)' if FULL_LEGEND else '_nolegend_',
                    color=COLOR_WALLETS_INPUTS, alpha=0.1, linewidth=1)
            ax.plot(predicted_wallets_inputs_avg,
                     label=f'Average predicted # wallets (inputs), window={LARGE_AVG_WINDOW}' if FULL_LEGEND else '_nolegend_',
                    color=COLOR_WALLETS_INPUTS, alpha=0.7, linewidth=1)
            if predicted_wallets_inputs_cilo_avg:
                # ax.fill_between(range(LARGE_AVG_WINDOW, len(predicted_wallets_inputs_cilo_avg) + LARGE_AVG_WINDOW), predicted_wallets_inputs_cilo_avg, predicted_wallets_inputs_cihi_avg, alpha=0.3, color=COLOR_WALLETS_INPUTS)
                ax.plot(predicted_wallets_inputs_cilo_avg,
                         label=f'Average predicted # wallets (inputs, CI lo), window={LARGE_AVG_WINDOW}' if FULL_LEGEND else '_nolegend_',
                        color=COLOR_WALLETS_INPUTS, alpha=0.3, linewidth=1, linestyle=':')
                ax.plot(predicted_wallets_inputs_cihi_avg,
                         label=f'Average predicted # wallets (inputs, CI hi), window={LARGE_AVG_WINDOW}' if FULL_LEGEND else '_nolegend_',
                        color=COLOR_WALLETS_INPUTS, alpha=0.3, linewidth=1, linestyle=':')

            # Artificial entry with same settings to have legend complete on ax
            ax.plot(predicted_wallets_list_inputs[0], label=f'Predicted # wallets (inputs, every coinjoin)',
                    color=COLOR_WALLETS_INPUTS, alpha=0.1, linewidth=1)
            ax.plot(predicted_wallets_inputs_avg[0],
                     label=f'Average predicted # wallets (inputs, window={LARGE_AVG_WINDOW})', color=COLOR_WALLETS_INPUTS, alpha=0.7, linewidth=1)

        if plot_outputs_prediction:
            ax.plot(predicted_wallets_list_outputs,
                     label=f'Predicted # wallets (outputs)' if FULL_LEGEND else '_nolegend_',
                    color=COLOR_WALLETS_OUTPUTS, alpha=0.1, linewidth=1)
            ax.plot(predicted_wallets_outputs_avg,
                     label=f'Average predicted # wallets (outputs), window={LARGE_AVG_WINDOW}' if FULL_LEGEND else '_nolegend_',
                    color=COLOR_WALLETS_OUTPUTS, alpha=0.7, linewidth=1)
            if predicted_wallets_outputs_cilo_avg:
                # ax.fill_between(range(LARGE_AVG_WINDOW, len(predicted_wallets_outputs_cilo_avg) + LARGE_AVG_WINDOW), predicted_wallets_outputs_cihi_avg, predicted_wallets_outputs_cilo_avg, color=COLOR_WALLETS_OUTPUTS, alpha=0.3)
                ax.plot(predicted_wallets_outputs_cilo_avg,
                         label=f'Average predicted # wallets (outputs, CI lo), window={LARGE_AVG_WINDOW}' if FULL_LEGEND else '_nolegend_',
                        color=COLOR_WALLETS_OUTPUTS, alpha=0.3, linewidth=1, linestyle=':')
                ax.plot(predicted_wallets_outputs_cihi_avg,
                         label=f'Average predicted # wallets (outputs, CI hi), window={LARGE_AVG_WINDOW}' if FULL_LEGEND else '_nolegend_',
                        color=COLOR_WALLETS_OUTPUTS, alpha=0.3, linewidth=1, linestyle=':')

            # Artificial entry with same settings to have legend complete on ax
            ax.plot(predicted_wallets_list_outputs[0], label=f'Predicted # wallets (outputs, every coinjoin)',
                    color=COLOR_WALLETS_OUTPUTS, alpha=0.1, linewidth=1)
            ax.plot(predicted_wallets_outputs_avg[0],
                     label=f'Average predicted # wallets (outputs, window={LARGE_AVG_WINDOW})', color=COLOR_WALLETS_OUTPUTS, alpha=0.7, linewidth=1)

        if predicted_wallets_inputs_cilo_avg:
            # ax.fill_between(range(LARGE_AVG_WINDOW, len(predicted_wallets_inputs_cilo_avg) + LARGE_AVG_WINDOW), predicted_wallets_inputs_cilo_avg, predicted_wallets_inputs_cihi_avg, alpha=0.3, color=COLOR_WALLETS_INPUTS)
            ax.plot(predicted_wallets_inputs_cilo_avg[0],
                     label=f'Confidence interval 5% - 95%',
                    color='gray', alpha=0.3, linewidth=1, linestyle=':')

        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.set_ylabel('number of predicted wallets')
        #ax.tick_params(axis='y', colors=COLOR_WALLETS_INPUTS)


    PLOT_INPUTS2OUTPUTS_FACTOR = False
    if PLOT_INPUTS2OUTPUTS_FACTOR and do_plot:
        ax2 = ax.twinx()

        ax2.plot(ratios_list_every_cjtx, label=f'Inputs/outputs-based factor (every coinjoin)', alpha=0.3, color='black')
        ax2.plot(ratios_list, label=f'L1 minimization, window={WINDOW_LEN}', alpha=0.5, color='black')
        ax2.plot(range(LARGE_AVG_WINDOW, len(ratios_list_avg) + LARGE_AVG_WINDOW), ratios_list_avg,
             label=f'Average of L1 minimization, window={LARGE_AVG_WINDOW}', color='red', alpha=0.5, linewidth=2)
        ax2.set_ylabel('inputs prediction factor')
        ax2.yaxis.set_label_position("left")
        ax2.yaxis.tick_left()

    save_path = os.path.join(target_load_path, f'{mix_id}_wallets_predictions_dynamics')
    # Finalize graph
    if do_plot:
        plt.subplots_adjust(bottom=0.15)
        ax.set_title(f'Number of predicted participating wallets: {mix_id}')
        ax.legend(loc='upper left')
        plt.savefig(f'{save_path}.png', dpi=300)
        plt.savefig(f'{save_path}.pdf', dpi=300)
        logging.info(f'estimate_wallet_prediction_factor() saved into {save_path}.png')
    else:
        logging.info(f'No plotting requested')

    if ax_provided is None:
        plt.close()

    predicted_wallets = {}
    last_usable_factor = used_prediction_ratios[len(used_prediction_ratios) - 1]
    for i in range(0, len(sorted_cj_time)):
        predicted_wallets[sorted_cj_time[i]['txid']] = {'txid': sorted_cj_time[i]['txid'],
                                  'num_wallets': predicted_wallets_list_outputs[i],
                                  'num_wallets_by_inputs': predicted_wallets_list_inputs[i],
                                  'num_wallets_by_inputs_cilo': predicted_wallets_list_inputs_cilo[i] if predicted_wallets_list_inputs_cilo else None,
                                  'num_wallets_by_inputs_cihi': predicted_wallets_list_inputs_cihi[i] if predicted_wallets_list_inputs_cihi else None,
                                  'num_wallets_by_outputs': predicted_wallets_list_outputs[i],
                                  'num_wallets_by_outputs_cilo': predicted_wallets_list_outputs_cilo[i] if predicted_wallets_list_outputs_cilo else None,
                                  'num_wallets_by_outputs_cihi': predicted_wallets_list_outputs_cihi[i] if predicted_wallets_list_outputs_cihi else None,
                                  'separate_ctx_input_factor': list_get(ratios_list_every_cjtx, i, -1),
                                  f'L1_{WINDOW_LEN}_input_factor': list_get(ratios_list, i, last_usable_factor),
                                  f'L1_{WINDOW_LEN}_avg_{LARGE_AVG_WINDOW}_input_factor': list_get(ratios_list_avg, i, last_usable_factor)
                                  }
    als.save_json_to_file_pretty(f'{save_path}.json', {'mix_id':mix_id, 'predictions': predicted_wallets})

    return predicted_wallets, predicted_wallets_inputs_avg, predicted_wallets_outputs_avg


def plot_flows_steamgraph(flow_in_year: dict, title: str):
    start_year = 2019
    end_year = 2024

    flow_types = sorted([flow_type for flow_type in flow_in_year.keys()])

    COLORS = ['gray', 'green', 'olive', 'black', 'red', 'orange']
    fig, ax = plt.subplots(figsize=(10, 5))
    # end_year in x_axis must be + 1 to correct for 0 index in flow_data
    x_axis = np.linspace(start_year, end_year + 1, num=(end_year - start_year + 1) * 12)

    DRAW_WW1_WW2_FLOW = False
    if DRAW_WW1_WW2_FLOW:
        flow_data_1 = []
        flow_types_process_1 = ['Wasabi -> Wasabi2']
        for flow_type in flow_types_process_1:
            case_data = [round(flow_in_year[flow_type][year][month] / SATS_IN_BTC, 2) for year in flow_in_year[flow_type].keys()
                         for month in range(1, 13)]
            flow_data_1.append(case_data)
        ax.stackplot(x_axis, flow_data_1, labels=list(flow_types_process_1), colors=COLORS, baseline="sym", alpha=0.4)

    if DRAW_WW1_WW2_FLOW:
        flow_types_process_2 = [item for item in flow_types if item != 'Wasabi -> Wasabi2']
    else:
        flow_types_process_2 = [item for item in flow_types]
    flow_data_2 = []
    flow_data_labels_2 = []
    for flow_type in flow_types_process_2:
        case_data = [round(flow_in_year[flow_type][year][month] / SATS_IN_BTC, 2) for year in flow_in_year[flow_type].keys() for month in range(1, 13)]
        flow_data_2.append(case_data)
        flow_data_labels_2.append(f'{flow_type} ({sum(case_data)} btc)')
        assert len(case_data) == (end_year - start_year + 1) * 12
    if DRAW_WW1_WW2_FLOW:
        ax.stackplot(x_axis, flow_data_2, labels=flow_data_labels_2, colors=COLORS[1:], baseline="sym", alpha=0.7)
    else:
        ax.stackplot(x_axis, flow_data_2, labels=flow_data_labels_2, colors=COLORS, baseline="sym", alpha=0.7)

    ax.legend(loc="lower left")
    ax.set_title(title)
    #ax.set_yscale('log')  # If enabled, it does not plot correctly, possibly bug in mathplotlib
    plt.show()

    PLOT_SMOOTH = False
    if PLOT_SMOOTH:
        def gaussian_smooth(x, y, grid, sd):
            weights = np.transpose([stats.norm.pdf(grid, m, sd) for m in x])
            weights = weights / weights.sum(0)
            return (weights * y).sum(1)

        fig, ax = plt.subplots(figsize=(10, 5))
        grid = np.linspace(start_year, end_year + 1, num=1000)
        y_smoothed = [gaussian_smooth(x_axis, y_, grid, 0.05) for y_ in flow_data_1]
        ax.stackplot(grid, y_smoothed, labels=list(flow_types_process_1), colors=COLORS, baseline="sym", alpha=0.3)
        y_smoothed = [gaussian_smooth(x_axis, y_, grid, 0.05) for y_ in flow_data_2]
        ax.stackplot(grid, y_smoothed, labels=list(flow_types_process_2), colors=COLORS[1:], baseline="sym", alpha=0.7)
        ax.legend(loc="lower left")
        ax.set_title(title)
        plt.show()


def plot_flows_dumplings(flows: dict):
    num_flow_types = len(flows.keys())
    start_year = 2018
    end_year = 2024
    # num_months = (end_year - start_year)*12
    x = np.arange(start_year, end_year, 12)  # (N,) array-like
    np.random.seed(42)
    y = [np.random.randint(0, 5, size=end_year - start_year) for _ in range(num_flow_types)]

    flow_in_year = {}
    for flow_type in flows.keys():
        flow_in_year[flow_type] = {}
        for year in range(start_year, end_year + 1):
            flow_in_year[flow_type][year] = {}
            for month in range(1, 12 + 1):
                flow_in_year[flow_type][year][month] = sum(
                    [flows[flow_type][txid]['value'] for txid in flows[flow_type].keys()
                     if precomp_datetime.strptime(flows[flow_type][txid]['broadcast_time'],
                                                  "%Y-%m-%d %H:%M:%S.%f").year == year and
                     precomp_datetime.strptime(flows[flow_type][txid]['broadcast_time'],
                                               "%Y-%m-%d %H:%M:%S.%f").month == month
                     ])
    def gaussian_smooth(x, y, grid, sd):
        weights = np.transpose([stats.norm.pdf(grid, m, sd) for m in x])
        weights = weights / weights.sum(0)
        return (weights * y).sum(1)

    flow_data = []
    for flow_type in flows.keys():
        case_data = [flow_in_year[flow_type][year][month] for year in flow_in_year[flow_type].keys() for month in range(1, 13)]
        flow_data.append(case_data)
        assert len(case_data) == (end_year - start_year + 1) * 12
    #COLORS = sns.color_palette("twilight_shifted", n_colors=len(flow_data))
    #COLORS = sns.color_palette("RdYlGn", n_colors=len(flow_data))
    COLORS = ['red', 'orange', 'green', 'olive', 'gray', 'black']

    fig, ax = plt.subplots(figsize=(10, 5))
    # end_year in x_axis must be + 1 to correct for 0 index in flow_data
    x_axis = np.linspace(start_year, end_year + 1, num=(end_year - start_year + 1) * 12)
    ax.stackplot(x_axis, flow_data, labels=list(flows.keys()), colors=COLORS, baseline="sym", alpha=1)
    ax.legend(loc="lower left")
    #ax.set_yscale('log')  # If enabled, it does not plot correctly, possibly bug in mathplotlib
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    grid = np.linspace(start_year - 1, end_year + 1, num=500)
    y_smoothed = [gaussian_smooth(x, y_, grid, 0.1) for y_ in flow_data]
    ax.stackplot(grid, y_smoothed, labels=list(flows.keys()), colors=COLORS, baseline="sym", alpha=0.7)

    ax.legend()
    plt.show()


def plot_steamgraph_example():
    x = np.arange(1990, 2020)  # (N,) array-like
    y = [np.random.randint(0, 5, size=30) for _ in range(5)]  # (M, N) array-like

    def gaussian_smooth(x, y, grid, sd):
        weights = np.transpose([stats.norm.pdf(grid, m, sd) for m in x])
        weights = weights / weights.sum(0)
        return (weights * y).sum(1)

    COLORS = ["#D0D1E6", "#A6BDDB", "#74A9CF", "#2B8CBE", "#045A8D"]
    fig, ax = plt.subplots(figsize=(10, 7))
    grid = np.linspace(1985, 2025, num=500)
    y_smoothed = [gaussian_smooth(x, y_, grid, 1) for y_ in y]
    ax.stackplot(grid, y_smoothed, colors=COLORS, baseline="sym")
    plt.show()


def plot_zksnacks_output_clusters(target_path: str, mix_id: str, sorted_output_nums:dict, sorted_output_distrib: dict):
    plt.figure(figsize=(10, 3))
    # plt.bar(list(sorted_input_distrib.keys()), list(sorted_input_distrib.values()), color='red', alpha=0.4, label='Input wallet clusters')
    # plt.plot(list(sorted_input_nums.keys()), list(sorted_input_nums.values()), color='red', alpha=1, label='Number of inputs')
    plt.plot(list(sorted_output_nums.keys()), list(sorted_output_nums.values()), color='royalblue', alpha=0.8,
             label='Number of outputs')
    plt.bar(list(sorted_output_distrib.keys()), list(sorted_output_distrib.values()), color='royalblue',
            alpha=0.6, label='Output wallet clusters')
    plt.title(f'{mix_id}: distribution of number of distinct output clusters per each coinjoin')
    plt.xlabel(f'Number of clusters / inputs / outputs')
    plt.ylabel(f'Number of occurences')
    plt.legend()
    save_file = os.path.join(target_path, mix_id, f'{mix_id}_distinct_wallets_output_zksnacks')
    plt.subplots_adjust(bottom=0.17)
    plt.savefig(f'{save_file}.png', dpi=300)
    plt.savefig(f'{save_file}.pdf', dpi=300)
    plt.close()


def plot_inputs_distribution(mix_id, inputs):
    log_data = np.log(inputs)
    hist, bins = np.histogram(log_data, bins=100)
    plt.bar(bins[:-1], hist, width=np.diff(bins))
    xticks = np.linspace(min(log_data), max(log_data), num=10)
    plt.xscale('log')
    plt.xticks(xticks, np.round(np.exp(xticks), decimals=0), rotation=45, fontsize=6)
    plt.title(f'{mix_id} inputs histogram (x axis is log)')
    plt.xlabel(f'Size of input')
    plt.ylabel(f'Number of inputs')
    plt.show()



def plot_inputs_type_ratio(mix_id: str, data: dict, initial_cj_index: int, ax, analyze_values: bool, normalize_values: bool, restrict_to_in_size: (int, int) = None):
    """
    Ratio between various types of inputs (fresh, remixed, remixed_friends)
    :param mix_id:
    :param data:
    :param ax:
    :param analyze_values if true, then size of inputs is analyzed, otherwise only numbers
    :return:
    """
    logging.info(f'plot_inputs_type_ratio(mix_id={mix_id}, analyze_values={analyze_values}, normalize_values={normalize_values})')

    coinjoins = data['coinjoins']
    sorted_cj_time = als.sort_coinjoins(coinjoins, als.SORT_COINJOINS_BY_RELATIVE_ORDER)
    #sorted_cj_time = sorted_cj_time[0:500]

    if restrict_to_in_size is None:
        restrict_to_in_size = (0, 1000000000000)
        print(f'No limits for inputs value')
    else:
        print(f'Limits for inputs value is {restrict_to_in_size[0]} - {restrict_to_in_size[1]}')

    input_types_nums = {}
    for event_type in MIX_EVENT_TYPE:
        if analyze_values:
            # Sum of values of inputs is taken
            input_types_nums[event_type.name] = [sum([coinjoins[cjtx['txid']]['inputs'][index]['value'] for index in coinjoins[cjtx['txid']]['inputs'].keys()
                                        if coinjoins[cjtx['txid']]['inputs'][index]['mix_event_type'] == event_type.name and
                                                      restrict_to_in_size[0] <= coinjoins[cjtx['txid']]['inputs'][index]['value'] <= restrict_to_in_size[1]])
                                            for cjtx in sorted_cj_time]
        else:
            # Only number of inputs is taken
            input_types_nums[event_type.name] = [sum([1 for index in coinjoins[cjtx['txid']]['inputs'].keys()
                                        if coinjoins[cjtx['txid']]['inputs'][index]['mix_event_type'] == event_type.name and
                                                      restrict_to_in_size[0] <= coinjoins[cjtx['txid']]['inputs'][index]['value'] <= restrict_to_in_size[1]])
                                   for cjtx in sorted_cj_time]

    # Obtain vector number of inputs/values for each remix, based on burn time
    # First take remixes with standard denominations
    event_type = MIX_EVENT_TYPE.MIX_REMIX
    BURN_TIME_RANGES = [('1', 1, 1), ('2', 2, 2), ('3-5', 3, 5), ('6-19', 6, 19), ('20+', 20, 999), ('1000-1999', 1000, 1999), ('2000+', 2000, 1000000)]
    for range_val in BURN_TIME_RANGES:
        input_types_nums[f'{event_type.name}_{range_val[0]}'] = als.get_inputs_type_list(coinjoins, sorted_cj_time, event_type, 'inputs', range_val[1], range_val[2], analyze_values, restrict_to_in_size, True)
    # Add remixes of non-standard denominations ("change" outputs)
    input_types_nums['MIX_REMIX_nonstd'] = als.get_inputs_type_list(coinjoins, sorted_cj_time, MIX_EVENT_TYPE.MIX_REMIX, 'inputs', 1, 10000000, analyze_values, restrict_to_in_size, False)

    short_exp_name = mix_id

    # Normalize all values into range 0-1 (only MIX_ENTER, MIX_REMIX and MIX_REMIX_FRIENDS are considered for base total)
    input_types_nums_normalized = {}
    total_values = (np.array(input_types_nums[MIX_EVENT_TYPE.MIX_ENTER.name]) + np.array(input_types_nums[MIX_EVENT_TYPE.MIX_REMIX.name]) +
                    np.array(input_types_nums[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name]) + np.array(input_types_nums[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS_WW1.name]))
    # Normalize all values including 'MIX_REMIX_1-2' etc.
    for item in input_types_nums.keys():
        input_types_nums_normalized[item] = np.array(input_types_nums[item]) / total_values

    def print_inputs_stats(input_types: dict, start_offset: int = 0, end_offset: int = -1):
        logging.info(f'  MIX_ENTER median ratio: {round(np.median(input_types[MIX_EVENT_TYPE.MIX_ENTER.name][start_offset: end_offset]) * 100, 2)}%')
        logging.info(f'  MIX_REMIX_nonstd median ratio: {round(np.median(input_types["MIX_REMIX_nonstd"][start_offset: end_offset]) * 100, 2)}%')
        logging.info(f'  MIX_REMIX median ratio: {round(np.median(input_types[MIX_EVENT_TYPE.MIX_REMIX.name][start_offset: end_offset]) * 100, 2)}%')
        for range_val in BURN_TIME_RANGES:
            remix_name = f'{event_type.name}_{range_val[0]}'
            print(f'  {remix_name} median ratio: {round(np.median(input_types[remix_name][start_offset: end_offset]) * 100, 2)}%')
        logging.info(f'  MIX_REMIX_FRIENDS median ratio: {round(np.median(input_types[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name][start_offset: end_offset]) * 100, 2)}%')
        logging.info(f'  MIX_REMIX_FRIENDS_WW1 median ratio: {round(np.median(input_types[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS_WW1.name][start_offset: end_offset]) * 100, 2)}%')

    logging.info(f' Inputs ratios [all]')
    print_inputs_stats(input_types_nums_normalized)
    logging.info(f' Inputs ratios [skip first two]')
    print_inputs_stats(input_types_nums_normalized, 2, -1)
    logging.info(f' Inputs ratios [skip first five]')
    print_inputs_stats(input_types_nums_normalized, 5, -1)

    # Convert non-normalized values from sats to btc (for sats values only)
    if analyze_values:
        for item in input_types_nums.keys():
            input_types_nums[item] = np.array(input_types_nums[item]) / SATS_IN_BTC

    # Set normalized or non-normalized version to use
    input_types = input_types_nums_normalized if normalize_values else input_types_nums

    bar_width = 0.3
    categories = range(0, len(sorted_cj_time))

    # New version with separated remixes
    bars = []
    bars.append((input_types[MIX_EVENT_TYPE.MIX_ENTER.name], 'MIX_ENTER', 'blue', 0.9))
    bars.append((input_types['MIX_REMIX_nonstd'], 'MIX_REMIX_nonstd', 'blue', 0.3))
    bars.append((input_types['MIX_REMIX_1'], 'MIX_REMIX_1', 'gold', 0.8))
    bars.append((input_types['MIX_REMIX_2'], 'MIX_REMIX_2', 'orange', 0.4))
    bars.append((input_types['MIX_REMIX_3-5'], 'MIX_REMIX_3-5', 'orange', 0.8))
    bars.append((input_types['MIX_REMIX_6-19'], 'MIX_REMIX_6-19', 'moccasin', 0.5))
    bars.append((input_types['MIX_REMIX_20+'], 'MIX_REMIX_20+', 'lightcoral', 0.7))
    bars.append((input_types['MIX_REMIX_1000-1999'], 'MIX_REMIX_1000-1999', 'sienna', 0.7))
    bars.append((input_types['MIX_REMIX_2000+'], 'MIX_REMIX_2000+', 'sienna', 1))
    bars.append((input_types[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name], 'MIX_REMIX_FRIENDS', 'green', 0.5))
    bars.append((input_types[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS_WW1.name], 'MIX_REMIX_FRIENDS_WW1', 'green', 0.9))

    # Draw all inserted bars atop each other
    if ax:
        bar_bottom = None
        for bar_item in bars:
            if bar_bottom is None:
                ax.bar(categories, bar_item[0], bar_width, label=f'{bar_item[1]} {short_exp_name}', alpha=bar_item[3],
                       color=bar_item[2], linewidth=0)
                bar_bottom = np.array(bar_item[0])
            else:
                ax.bar(categories, bar_item[0], bar_width, label=f'{bar_item[1]} {short_exp_name}', alpha=bar_item[3], color=bar_item[2],
                        bottom=bar_bottom, linewidth=0)
                bar_bottom = bar_bottom + np.array(bar_item[0])

        ax.set_title(f'Type of inputs for given cjtx ({"values" if analyze_values else "number"})\n{short_exp_name}')
        ax.set_xlabel('Coinjoin in time')
        if analyze_values and normalize_values:
            ax.set_ylabel('Fraction of inputs values')
        if analyze_values and not normalize_values:
            ax.set_ylabel('Inputs values (btc)', fontsize=DEFAULT_AXIS_LABEL_SIZE)
        if not analyze_values and normalize_values:
            ax.set_ylabel('Fraction of input numbers')
        if not analyze_values and not normalize_values:
            ax.set_ylabel('Number of inputs')

    PLOT_REMIX_RATIO = False
    if PLOT_REMIX_RATIO:
        WINDOWS_SIZE = 1
        remix_ratios_all = [input_types[MIX_EVENT_TYPE.MIX_REMIX.name][i] * 100 for i in range(len(input_types[MIX_EVENT_TYPE.MIX_REMIX.name]))]  # All remix including nonstandard
        remix_ratios_nonstd = [input_types['MIX_REMIX_nonstd'][i] * 100 for i in range(len(input_types['MIX_REMIX_nonstd']))]  # Nonstd remixes
        remix_ratios_std = [remix_ratios_all[i] - remix_ratios_nonstd[i] for i in range(len(remix_ratios_all))]  # Only standard remixes
        remix_ratios_avg = [np.average(remix_ratios_std[i:i+WINDOWS_SIZE]) for i in range(0, len(remix_ratios_std), WINDOWS_SIZE)]
        if ax:
            ax2 = ax.twinx()
            ax2.plot(range(0, len(remix_ratios_avg), WINDOWS_SIZE), remix_ratios_avg, label=f'MIX_REMIX avg({WINDOWS_SIZE})', color='brown', linewidth=1, linestyle='--', alpha=0.4)
            ax2.set_ylim(0, 100)  # Force whole range of yaxis
            ax2.tick_params(axis='y', colors='brown', labelsize=6)
            ax2.set_ylabel('Average remix rate %', color='brown', fontsize='6')

    return input_types


def plot_mix_liquidity(mix_id: str, data: dict, initial_liquidity, time_liquidity: dict, initial_cj_index: int, ax):
    coinjoins = data['coinjoins']
    sorted_cj_time = als.sort_coinjoins(coinjoins, als.SORT_COINJOINS_BY_RELATIVE_ORDER)

    # New fresh liquidity
    mix_enter = [sum([coinjoins[cjtx['txid']]['inputs'][index]['value'] for index in coinjoins[cjtx['txid']]['inputs'].keys()
                                if coinjoins[cjtx['txid']]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_ENTER.name])
                           for cjtx in sorted_cj_time]
    # Input liquidity from friends (one hop remix)
    mix_remixfriend = [sum([coinjoins[cjtx['txid']]['inputs'][index]['value'] for index in coinjoins[cjtx['txid']]['inputs'].keys()
                                if coinjoins[cjtx['txid']]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name])
                           for cjtx in sorted_cj_time]
    # Input liquidity from ww1
    mix_remixfriend_ww1 = [sum([coinjoins[cjtx['txid']]['inputs'][index]['value'] for index in coinjoins[cjtx['txid']]['inputs'].keys()
                                if coinjoins[cjtx['txid']]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX_FRIENDS_WW1.name])
                           for cjtx in sorted_cj_time]
    # Output spent outside mix
    mix_leave = [sum([coinjoins[cjtx['txid']]['outputs'][index]['value'] for index in coinjoins[cjtx['txid']]['outputs'].keys()
                                    if coinjoins[cjtx['txid']]['outputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_LEAVE.name])
                               for cjtx in sorted_cj_time]

    # Output staying in mix
    mix_stay = [sum([coinjoins[cjtx['txid']]['outputs'][index]['value'] for index in coinjoins[cjtx['txid']]['outputs'].keys()
                                    if coinjoins[cjtx['txid']]['outputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_STAY.name])
                               for cjtx in sorted_cj_time]

    INTERVAL_LENGTH = 3 * 30 * 24 * 3600  # 3 months == 3 * 30 * 24 * 3600
    INTERVAL_LENGTH = 30 * 24 * 3600  # 1 month == * 30 * 24 * 3600
    # Outputs leaving mix `fast` after its mixing (within 0-INTERVAL_LENGTH seconds)
    mix_leave_timecutoff_before = [sum([coinjoins[cjtx['txid']]['outputs'][index]['value'] for index in coinjoins[cjtx['txid']]['outputs'].keys()
                                    if coinjoins[cjtx['txid']]['outputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_LEAVE.name and
                                      coinjoins[cjtx['txid']]['outputs'][index]['burn_time'] < INTERVAL_LENGTH])
                               for cjtx in sorted_cj_time]
    COMPUTE_UNUSED = False
    if COMPUTE_UNUSED:
        # Outputs leaving mix `slow` after its mixing (at least after INTERVAL_LENGTH seconds)
        mix_leave_timecutoff_after = [sum([coinjoins[cjtx['txid']]['outputs'][index]['value'] for index in coinjoins[cjtx['txid']]['outputs'].keys()
                                        if coinjoins[cjtx['txid']]['outputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_LEAVE.name and
                                          coinjoins[cjtx['txid']]['outputs'][index]['burn_time'] >= INTERVAL_LENGTH])
                                   for cjtx in sorted_cj_time]

        # Output staying in mix MIX_EVENT_TYPE.MIX_REMIX
        mix_remix = [sum([coinjoins[cjtx['txid']]['outputs'][index]['value'] for index in coinjoins[cjtx['txid']]['outputs'].keys()
                                        if coinjoins[cjtx['txid']]['outputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX.name])
                                   for cjtx in sorted_cj_time]
        # Output staying in mix MIX_EVENT_TYPE.MIX_REMIX with non-standard values
        mix_remix_nonstandard = [sum([coinjoins[cjtx['txid']]['outputs'][index]['value'] for index in coinjoins[cjtx['txid']]['outputs'].keys()
                                        if coinjoins[cjtx['txid']]['outputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX.name and
                                           coinjoins[cjtx['txid']]['outputs'][index].get('is_standard_denom') == False])
                                   for cjtx in sorted_cj_time]


    cjtx_cummulative_liquidity = []
    cjtx_cummulative_liquidity_timecutoff = []
    curr_liquidity = initial_liquidity[0]  # Take last cummulative liquidity (MIX_ENTERxxx - MIX_LEAVE) from previous interval
    curr_liquidity_timecutoff = initial_liquidity[3]
    assert len(mix_enter) == len(mix_leave) == len(mix_remixfriend) == len(mix_remixfriend_ww1) == len(mix_stay), logging.error(f'Mismatch in length of input/out sum arrays: {len(mix_enter)} vs. {len(mix_leave)}')
    # Change in liquidity as observed by each coinjoin (increase directly when mix_enter, decrease directly even when mix_leave happens later in wall time)
    for index in range(0, len(mix_enter)):
        liquidity_step = mix_enter[index] + mix_remixfriend[index] + mix_remixfriend_ww1[index] - mix_leave[index]
        # Print significant changes in liquidity for easier debugging
        if mix_enter[index] > 100 * SATS_IN_BTC:
            print(f'Fresh input jump    of {round(mix_enter[index] / SATS_IN_BTC, 1)} at {index}: {sorted_cj_time[index]}')
        if liquidity_step > 100 * SATS_IN_BTC:
            print(f'Pool liquidity jump of {round(liquidity_step / SATS_IN_BTC, 1)} at {index}: {sorted_cj_time[index]}')
        curr_liquidity = curr_liquidity + liquidity_step
        cjtx_cummulative_liquidity.append(curr_liquidity)

        # Same computation, but assume as leaving only mix_leave_timecutoff value
        # time-limited value (< INTERVAL_LENGTH)
        curr_liquidity_timecutoff = curr_liquidity_timecutoff + liquidity_step + mix_leave[index] - mix_leave_timecutoff_before[index]
        cjtx_cummulative_liquidity_timecutoff.append(curr_liquidity_timecutoff)

    # Cumulative liquidity never remixed or leaving mix (MIX_STAY coins)
    stay_liquidity = []
    stay_liquidity_timecutoff = []
    curr_stay_liquidity = initial_liquidity[1]  # Take last cumulative liquidity (MIX_STAY) from previous interval
    curr_stay_liquidity_timecutoff = initial_liquidity[4]  # Take last cumulative liquidity (MIX_STAY) from previous interval
    for index in range(0, len(mix_stay)):
        curr_stay_liquidity = curr_stay_liquidity + mix_stay[index]
        stay_liquidity.append(curr_stay_liquidity)

        # time-limited value (=> INTERVAL_LENGTH)
        #curr_stay_liquidity_timecutoff = curr_stay_liquidity_timecutoff + mix_stay_timecutoff_before[index]
        #stay_liquidity_timecutoff.append(curr_stay_liquidity_timecutoff)

    # Remixed liquidity levels
    remix_liquidity = []
    curr_remix_liquidity = initial_liquidity[2]  # Take last remix liquidity from previous interval
    for index in range(0, len(mix_stay)):
        remix_liquidity_step = mix_enter[index] + mix_remixfriend[index] + mix_remixfriend_ww1[index] - mix_leave[index] - stay_liquidity[index]  # prev state + new input liquidity - output liqudity
        # BUGBUG: We must also consider exact evaluation of mining fee payed to get perfect match for the assert below
        #assert mix_remix[index] == remix_liquidity_step, f'Inconsistent remix liquidity estimation for {index}th coinjoin ({sorted_cj_time[index]['txid']}); Expected {mix_remix[index]} got {remix_liquidity_step[index]}'
        curr_remix_liquidity = curr_remix_liquidity + remix_liquidity_step
        remix_liquidity.append(curr_remix_liquidity)

    # Plot in btc
    liquidity_btc = [item / SATS_IN_BTC for item in cjtx_cummulative_liquidity]
    liquidity_timecutoff_btc = [item / SATS_IN_BTC for item in cjtx_cummulative_liquidity_timecutoff]
    stay_liquidity_btc = [item / SATS_IN_BTC for item in stay_liquidity]
    remix_liquidity_btc = [item / SATS_IN_BTC for item in remix_liquidity]
    if ax:
        #x_ticks = range(initial_cj_index, initial_cj_index + len(liquidity_btc))
        ax.plot(liquidity_btc, color='royalblue', alpha=0.6, linewidth=3)
        #ax.plot(stay_liquidity_btc, color='royalblue', alpha=0.6, linestyle='--')
        #ax.plot(remix_liquidity_btc, color='black', alpha=0.6, linestyle='--')
        PLOT_LEAVE_TIMECUTOFF = False
        if PLOT_LEAVE_TIMECUTOFF:
            ax.plot(liquidity_timecutoff_btc, color='blue', alpha=0.6, linestyle='--')

        ax.set_ylabel('btc in mix', color='royalblue', fontsize=DEFAULT_AXIS_LABEL_SIZE)
        ax.tick_params(axis='y', colors='royalblue')

    return cjtx_cummulative_liquidity, stay_liquidity, remix_liquidity, cjtx_cummulative_liquidity_timecutoff, stay_liquidity_timecutoff


def plot_mining_fee_rates(mix_id: str, data: dict, mining_fees: dict, ax):
    coinjoins = data['coinjoins']
    # Take real mining time as mining fee are more relevant to it, but adapt to relative ordering used for plotting
    sorted_cj_fee_time = als.sort_coinjoins(coinjoins, False)  # Real time of mining => time of minig fee rate application
    sorted_cj_fee_time_dict = {cj['txid']: cj for cj in sorted_cj_fee_time}  # Turn list into dict for faster lookups
    sorted_cj_time = als.sort_coinjoins(coinjoins, False)  # Take relative ordering of cjtxs

    # For each coinjoin find the closest fee rate record and plot it
    fee_rates = []
    fee_start_index = 0
    for cj in sorted_cj_time:
        timestamp = sorted_cj_fee_time_dict[cj['txid']]['broadcast_time'].timestamp()
        while timestamp > mining_fees[fee_start_index]['timestamp']:
            fee_start_index = fee_start_index + 1
            if fee_start_index >= len(mining_fees):
                logging.error(f'Missing mining_fees entry for timestamp {sorted_cj_fee_time_dict[cj["txid"]]["broadcast_time"]} if {cj["txid"]}.')
                # Use the latest one and stop searching
                fee_start_index = fee_start_index - 1
                break

        closest_fee = mining_fees[fee_start_index - 1]['avgFee_90']
        fee_rates.append(closest_fee)

    if ax:
        ax.plot(fee_rates, color='gray', alpha=0.4, linewidth=1, linestyle='--')
        ax.tick_params(axis='y', colors='gray', labelsize=6)
        ax.set_ylabel('Mining fee rate sats/vB (90th percentil)', color='gray', fontsize='6', labelpad=-2)

    return fee_rates



def plot_num_wallets(mix_id: str, data: dict, avg_input_ratio: dict, ax):
    coinjoins = data['coinjoins']
    sorted_cj_time = als.sort_coinjoins(coinjoins, als.SORT_COINJOINS_BY_RELATIVE_ORDER)

    if mix_id not in avg_input_ratio:
        avg_input_ratio['per_interval'][mix_id] = {}

    # Naive approach: For each coinjoin, compute as number of inputs divided by average inputs per wallet
    AVG_NUM_INPUTS, AVG_NUM_OUTPUTS = als.get_wallets_prediction_ratios(mix_id)

    # Find heuristically the AVG_NUM_INPUTS and AVG_NUM_OUTPUTS to minimize difference between computed number of inputs and outputs
    FIND_SYNTHETIC_RATIO = True
    if FIND_SYNTHETIC_RATIO:
        avg_input_ratio['factor_inputs_wallets'] = AVG_NUM_INPUTS
        avg_input_ratio['factor_outputs_wallets'] = AVG_NUM_OUTPUTS

        X = np.array([len(coinjoins[cj['txid']]['inputs']) for cj in sorted_cj_time])
        Y = np.array([len(coinjoins[cj['txid']]['outputs']) for cj in sorted_cj_time])

        # Objective function to minimize
        def objective_euclidean(params):
            x1, y1 = params
            return np.sum((X / x1 - Y / y1) ** 2)

        def objective_linear(params):
            x1, y1 = params
            return np.sum(np.abs(X / x1 - Y / y1))

        # Initial guess for x1 and y1
        initial_guess = [1, 1]
        # Minimize the objective function
        result = minimize(objective_linear, initial_guess, method='Nelder-Mead')
        # Optimal values
        x1_opt, y1_opt = result.x
        #AVG_NUM_OUTPUTS = AVG_NUM_INPUTS * (y1_opt / x1_opt)
        AVG_NUM_INPUTS = AVG_NUM_OUTPUTS * (x1_opt / y1_opt)
        print(f"Ratio y1/x1: {y1_opt / x1_opt}, Optimal x1: {x1_opt}, Optimal y1: {y1_opt}")
        print(f"AVG_NUM_OUTPUTS = {AVG_NUM_OUTPUTS} factor => AVG_NUM_INPUTS = {AVG_NUM_INPUTS} factor after scaling")
        avg_input_ratio['per_interval'][mix_id]['ratio_factor_outputs_inputs_wallets'] = y1_opt / x1_opt
        avg_input_ratio['per_interval'][mix_id]['factor_inputs_wallets'] = AVG_NUM_INPUTS
        avg_input_ratio['per_interval'][mix_id]['factor_outputs_wallets'] = AVG_NUM_OUTPUTS
        avg_input_ratio['all'].extend([AVG_NUM_INPUTS] * len(sorted_cj_time))

    num_wallets_naive_inputs = [len(coinjoins[cj['txid']]['inputs']) / AVG_NUM_INPUTS for cj in sorted_cj_time]
    num_wallets_naive_outputs = [len(coinjoins[cj['txid']]['outputs']) / AVG_NUM_OUTPUTS for cj in sorted_cj_time]

    # Load from other computed option
    num_wallets_predicted = [coinjoins[cj['txid']].get('num_wallets_predicted', -100) for cj in sorted_cj_time]
    # Set value for missing ones to nearby value
    last_val = 0
    for index in range(0, len(num_wallets_predicted)):
        if num_wallets_predicted[index] == -100:
            num_wallets_predicted[index] = last_val
        else:
            last_val = num_wallets_predicted[index]

    if ax:
        AVG_WINDOWS = 10
        #AVG_WINDOWS = 5
        COLOR_WALLETS_INPUTS = 'red'
        #COLOR_WALLETS_OUTPUTS = 'magenta'
        COLOR_WALLETS_OUTPUTS = 'green'
        num_wallets_avg_inputs = als.compute_averages(num_wallets_naive_inputs, AVG_WINDOWS)
        num_wallets_avg_inputs = np.array(num_wallets_avg_inputs)
        x = range(AVG_WINDOWS // 2, len(num_wallets_avg_inputs) + AVG_WINDOWS // 2)
        #x = range(AVG_WINDOWS, len(num_wallets_avg_inputs) + AVG_WINDOWS)
        #x = range(0, len(num_wallets_avg_inputs))
        ax.plot(x, num_wallets_avg_inputs, color=COLOR_WALLETS_INPUTS, alpha=0.4, linewidth=2, linestyle='-',
                label=f'Predicted wallets (inputs, avg={AVG_WINDOWS}, factor={round(AVG_NUM_INPUTS, 2)})')
        num_wallets_avg_outputs = als.compute_averages(num_wallets_naive_outputs, AVG_WINDOWS)
        num_wallets_avg_outputs = np.array(num_wallets_avg_outputs)
        ax.plot(x, num_wallets_avg_outputs, color=COLOR_WALLETS_OUTPUTS, alpha=0.4, linewidth=2, linestyle='-',
                label=f'Predicted wallets (outputs, avg={AVG_WINDOWS}, factor={round(AVG_NUM_OUTPUTS, 2)})')
        ax.tick_params(axis='y', colors=COLOR_WALLETS_INPUTS, labelsize=6)
        ax.fill_between(x, num_wallets_avg_inputs, num_wallets_avg_outputs, where=num_wallets_avg_inputs>num_wallets_avg_outputs, interpolate=True, color=COLOR_WALLETS_INPUTS, alpha=0.3)
        ax.fill_between(x, num_wallets_avg_inputs, num_wallets_avg_outputs, where=num_wallets_avg_outputs>num_wallets_avg_inputs, interpolate=True, color=COLOR_WALLETS_OUTPUTS, alpha=0.3)
        max_wallets_y = max(max(num_wallets_avg_outputs), max(num_wallets_avg_inputs))
        ax.set_yticks(np.arange(0, max_wallets_y + round(max_wallets_y * 0.1), step=10))

        #ax.set_ylabel('Estimated number of active wallets (naive)', color='red', fontsize='6')
        PLOT_WALLETS_PREDICTED = False
        if PLOT_WALLETS_PREDICTED:
            num_wallets_avg_predicted = als.compute_averages(num_wallets_predicted, AVG_WINDOWS)
            ax.plot(num_wallets_avg_predicted, color='green', alpha=0.4, linewidth=1, linestyle='-', label='Predicted wallets (model)')
            ax.tick_params(axis='y', colors='green', labelsize=6)
            ax.set_ylabel('Estimated number of active wallets (model)', color='green', fontsize='6')

        ax.legend()

    return num_wallets_predicted


def generate_liquidity_summary_html(coords: list, target_path: str):
    """
    Generate an embeddable HTML segment (no <html>/<head>/<body>) from a stats JSON.
    Now merges Inputs per CoinJoin and Outputs per CoinJoin into a single tab.
    """

    def fmt_btc(x):
        s = f"{float(x):.2f}".rstrip('0').rstrip('.')
        return s

    def safe(x):
        return escape(str(x)) if x is not None else ""

    def build_kv(key, value):
        return f'<span class="kv"><span class="k">{safe(key)}</span><span class="v">{safe(value)}</span></span>'

    def build_card(title, items):
        return f'''
        <div class="card">
          <div class="card-title">{escape(title)}</div>
          <div class="card-body">
            {''.join(items)}
          </div>
        </div>
        '''

    for coord in coords:
        pool_name = f'{coord[0]}_{coord[1]}' if len(coord[1]) > 0 else f'{coord[0]}'
        load_path = os.path.join(target_path, f'liquidity_summary_{pool_name}.json')
        print(f'Loading liquidity file {load_path}')
        data = als.load_json_from_file(load_path)
        if not data:
            print(f'  {load_path} not loaded, continuing...')
            continue

        earliest_time = data.get("earliest_time")
        latest_time = data.get("latest_time")
        earliest_cjtx = data.get("earliest_cjtx")
        latest_cjtx = data.get("latest_cjtx")
        total_coinjoins = data.get("total_coinjoins")

        min_inputs = data.get("min_inputs")
        max_inputs = data.get("max_inputs")
        avg_inputs = data.get("avg_inputs")
        median_inputs = data.get("median_inputs")

        min_outputs = data.get("min_outputs")
        max_outputs = data.get("max_outputs")
        avg_outputs = data.get("avg_outputs")
        median_outputs = data.get("median_outputs")

        total_fresh_inputs_value = data.get("total_fresh_inputs_value")
        total_friends_inputs_value = data.get("total_friends_inputs_value")
        total_unmoved_outputs_value = data.get("total_unmoved_outputs_value")
        total_leaving_outputs_value = data.get("total_leaving_outputs_value")
        total_nonstandard_leaving_outputs_value = data.get("total_nonstandard_leaving_outputs_value")
        total_fresh_inputs_without_nonstandard_outputs_value = data.get(
            "total_fresh_inputs_without_nonstandard_outputs_value")

        ratios_keys = [
            ("Fresh inputs / total inputs", data.get("ratio_fresh_inputs_2_total_inputs")),
            ("Friends inputs / total inputs", data.get("ratio_friends_inputs_2_total_inputs")),
            ("Leaving outputs / total outputs", data.get("ratio_leaving_outputs_2_total_outputs")),
            ("Staying outputs / total outputs", data.get("ratio_staying_outputs_2_total_outputs")),
            ("Staying / non-remix outputs", data.get("ratio_staying_outputs_2_nonremix_outputs")),
            ("Remixed inputs / total (numbers)", data.get("ratio_remixed_inputs_2_total_inputs_numbers")),
            ("Remixed inputs / total (values, BTC)", data.get("ratio_remixed_inputs_2_total_inputs_values")),
        ]

        period = ""
        if earliest_time and latest_time:
            #period = f"{safe(earliest_time)} — {safe(latest_time)}"
            period = f'{safe(earliest_time)} <a href="https://mempool.space/tx/{earliest_cjtx}">(tx)</a> — {safe(latest_time)} <a href="https://mempool.space/tx/{latest_cjtx}">(tx)</a>'

        # <h2>{coord[0]} — {coord[1]}</h2>
        header_html = f'''
        <div class="cjseg-header">
          <div class="meta">Total coinjoins: <b>{safe(total_coinjoins)}</b></div>
          <div class="meta">Period         : <b>{period}</b></div>
        </div>
        '''

        # Build the combined IO panel (single tab)
        inputs_line = f"{safe(min_inputs)} / {safe(round(avg_inputs, 1) if isinstance(avg_inputs, (int, float)) else avg_inputs)} / {safe(median_inputs)} / {safe(max_inputs)}"
        outputs_line = f"{safe(min_outputs)} / {safe(round(avg_outputs, 1) if isinstance(avg_outputs, (int, float)) else avg_outputs)} / {safe(median_outputs)} / {safe(max_outputs)}"

        io_panel_inner = f'''
          <div class="io-grid">
            <div class="io-col">
              <div class="over">Inputs per CJ</div>
              <div class="mono">{inputs_line}</div>
              <div class="under">min / avg / median / max</div>
            </div>
            <div class="io-col">
              <div class="over">Outputs per CJ</div>
              <div class="mono">{outputs_line}</div>
              <div class="under">min / avg / median / max</div>
            </div>
          </div>
        '''

        tabs_html = f'''
        <div class="tabs" role="tablist" aria-label="Stats tabs">
          <button class="tab active" role="tab" aria-selected="true" tabindex="0">Inputs &amp; Outputs</button>
        </div>
        <div class="tab-panels">
          <div class="tab-panel show" role="tabpanel">
            {io_panel_inner}
          </div>
        </div>
        '''

        ratio_items = [build_kv(k, v) for k, v in ratios_keys if v]
        ratio_card = build_card("Key Ratios", ratio_items)

        totals = []
        if total_fresh_inputs_value is not None:
            totals.append(build_kv("Fresh inflows (BTC)", fmt_btc(total_fresh_inputs_value)))
        if total_friends_inputs_value is not None:
            if total_fresh_inputs_without_nonstandard_outputs_value is not None:
                totals.append(build_kv("Fresh inflows without nonstandard (BTC)",
                                       fmt_btc(total_fresh_inputs_without_nonstandard_outputs_value)))
            totals.append(build_kv("WW1/WW2 mix inflows (BTC)", fmt_btc(total_friends_inputs_value)))
        if total_unmoved_outputs_value is not None:
            totals.append(build_kv("Unmoved outputs (BTC)", fmt_btc(total_unmoved_outputs_value)))
        if total_leaving_outputs_value is not None:
            totals.append(build_kv("Leaving outputs (BTC)", fmt_btc(total_leaving_outputs_value)))
        if total_nonstandard_leaving_outputs_value is not None:
            totals.append(build_kv("Nonstandard outputs leaving (BTC)", fmt_btc(total_nonstandard_leaving_outputs_value)))

        totals_card = build_card("Aggregate Values", totals) if totals else ""

        css = '''
        <style>
        .cjseg { --fg:#111; --bg:#fff; --muted:#666; --line:#e5e7eb; --accent:#111; font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji"; color:var(--fg);}
        .cjseg .cjseg-header { margin-bottom: 12px; }
        .cjseg h2 { margin: 0; font-size: 1.35rem; font-weight: 700; line-height: 1.3; }
        .cjseg .sub { color: var(--muted); margin-top: 2px; font-size: 0.9rem; }
        .cjseg .meta { margin-top: 6px; font-size: 0.95rem; }
        .cjseg .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px,1fr)); gap: 10px; }

        /* Cards */
        .cjseg .card { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: var(--bg); }
        .cjseg .card-title { font-weight: 600; margin-bottom: 6px; }
        .cjseg .card-body { display: grid; gap: 4px; }
        .cjseg .kv { display: flex; justify-content: space-between; gap: 10px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 0.92rem; }
        .cjseg .kv .k { color: var(--muted); }
        .cjseg .kv .v { font-weight: 600; }

        /* Tabs (single) */
        .cjseg .tabs { display: flex; gap: 6px; border-bottom: 1px solid var(--line); margin: 10px 0 8px; }
        .cjseg .tab { background: transparent; border: none; padding: 6px 8px; font-weight: 600; cursor: default; color: var(--fg); border-bottom: 2px solid var(--fg); }
        .cjseg .tab-panels { margin-top: 4px; }
        .cjseg .tab-panel { display: none; }
        .cjseg .tab-panel.show { display: block; }

        /* IO panel */
        .cjseg .io-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap: 10px; }
        .cjseg .io-col { border: 1px solid var(--line); border-radius: 8px; padding: 10px; }
        .cjseg .io-col .over { font-weight: 600; margin-bottom: 2px; }
        .cjseg .io-col .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 0.98rem; }
        .cjseg .io-col .under { color: var(--muted); font-size: 0.85rem; margin-top: 2px; }

        .cjseg .foot { margin-top: 8px; color: var(--muted); font-size: 0.85rem; }
        </style>
        '''

        html = f'''
        <section class="cjseg" aria-label="Liquidity summary">
          {css}
          {header_html}

          {tabs_html}

          <div class="grid">
            {ratio_card}
            {totals_card}
          </div>

        </section>
        '''.strip()
        #          <div class="foot">Source: coordinator logs. Segment is self-contained and embeddable.</div>

        out_path = os.path.join(target_path, f'{pool_name}.html')
        with open(out_path, "w", encoding="utf-8") as out:
            SM.print(f'Saving html export at {out_path}')
            out.write(html)


def plot_intermix_ratios(intercoord_ratios: dict, target_path: str | Path, prefix: str):
    for coordinator, records in intercoord_ratios.items():
        if len(records) == 0:
            continue
        # Convert nested dict into DataFrame
        df = pd.DataFrame.from_dict(records, orient="index")
        # Parse timestamps and sort
        df["broadcast_time"] = pd.to_datetime(df["broadcast_time"])
        df = df.sort_values("broadcast_time").reset_index(drop=True)

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(df["broadcast_time"], df["out_ratio"] * 100, label="outputs (same coordinator)", color='green', alpha=0.2)
        plt.plot(df["broadcast_time"], df["in_ratio"] * 100, label="inputs (same coordinator)", color='green', alpha=0.7)
        plt.plot(df["broadcast_time"], df["out_ratio_second"] * 100, label="outputs (second coordinator)", color='red', alpha=0.2)
        plt.plot(df["broadcast_time"], df["in_ratio_second"] * 100, label="inputs (second coordinator)", color='red', alpha=0.7)
        plt.axhline(40, color="gray", linestyle="--", linewidth=1, label="threshold (40%)")
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("% remix fraction")
        plt.title(f"'{coordinator}': intermixed ratio of inputs & outputs over time")
        plt.legend()
        plt.tight_layout()

        plt.savefig(Path(target_path, f"{prefix}in_out_ratio_over_time__{coordinator}.png"), dpi=200, bbox_inches="tight")
        #plt.savefig(Path(target_path, f"in_out_ratio_over_time__{coordinator}.pdf"), dpi=200, bbox_inches="tight")
        plt.close()

        coordinators = []
        in_series = []
        out_series = []
        for coordinator, records in intercoord_ratios.items():
            df = pd.DataFrame.from_dict(records, orient="index")
            in_vals = pd.to_numeric(df["in_ratio"],
                                    errors="coerce").dropna().tolist() if "in_ratio" in df.columns else []
            out_vals = pd.to_numeric(df["out_ratio"],
                                     errors="coerce").dropna().tolist() if "out_ratio" in df.columns else []
            if len(in_vals) == 0 and len(out_vals) == 0:
                continue
            coordinators.append(coordinator)
            in_series.append(in_vals if len(in_vals) > 0 else [float("nan")])
            out_series.append(out_vals if len(out_vals) > 0 else [float("nan")])

        M = len(coordinators)
        if M == 0:
            raise RuntimeError("No coordinators with in_ratio/out_ratio data found.")

        base_positions = list(range(M))
        offset = 0.15
        in_positions = [bp - offset for bp in base_positions]
        out_positions = [bp + offset for bp in base_positions]

        plt.figure(figsize=(max(8, M * 0.9), 3))

        bp_in = plt.boxplot(in_series, whis=(5,95), positions=in_positions, widths=0.25, patch_artist=True, showfliers=False)
        for patch in bp_in["boxes"]:
            patch.set(facecolor="#f28e2b")
        for element in ["whiskers", "caps", "medians"]:
            for line in bp_in[element]:
                line.set(color="#6b6b6b", linewidth=1.2)

        bp_out = plt.boxplot(out_series, whis=(5,95), positions=out_positions, widths=0.25, patch_artist=True, showfliers=False)
        for patch in bp_out["boxes"]:
            patch.set(facecolor="#4e79a7")

        # Rotate labels 45 degrees
        plt.xticks(base_positions, coordinators, rotation=15, ha="right")

        for element in ["whiskers", "caps", "medians"]:
            for line in bp_out[element]:
                line.set(color="#6b6b6b", linewidth=1.2)

        plt.xticks(base_positions, coordinators)
        plt.ylabel("ratio")
        plt.title("Ratio of intermixed inputs and outputs under same coordinator")

        # Add dashed horizontal line at 0.4
        plt.axhline(0.4, color="gray", linestyle="--", linewidth=1)

        legend_handles = [bp_in["boxes"][0], bp_out["boxes"][0]]
        plt.legend(legend_handles, ["inputs", "outputs"], loc="lower left")

        plt.tight_layout()
        plt.savefig(Path(target_path, f"{prefix}all_coordinators_in_out_boxplot.png"), dpi=200, bbox_inches="tight")
        plt.close()


def plot_coord_attribution_stats(main_coordinator: str, num_true_coord_txs: int, results: dict, target_path: str | Path, fp_string: str, fn_string: str, filename: str):
    def _to_float_or_die(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"Non-numeric key in results: {v!r}")

    # Build (numeric_value, original_key) pairs and sort by the numeric value
    _key_pairs = sorted(((_to_float_or_die(k), k) for k in results.keys()),
                        key=lambda t: t[0])

    # Use numeric values for the x-axis
    x_vals = [num for num, _orig in _key_pairs]

    # Use the original (unmodified) key to access the dict
    first_original_key = _key_pairs[0][1]
    coordinators = sorted(results[first_original_key].keys())

    # Build series dict: {coordinator: {"fp": [...], "fn": [...]}} aligned to x_vals
    series = {c: {fp_string: [], fn_string: []} for c in coordinators}

    for num, _orig in _key_pairs:
        x_str = _orig
        for c in coordinators:
            # Remove offset given by very first value where 0% changes were applied
            REMOVE_OFFSET = False
            if REMOVE_OFFSET:
                fp_offset = results["0"][c][fp_string][0]
                fn_offset = results["0"][c][fn_string][0]

            # Values are lists (single-element), extract safely with defaults
            entry = results[x_str].get(c, {})
            fp_list = entry.get(fp_string, [0])
            fn_list = entry.get(fn_string, [0])
            if REMOVE_OFFSET:
                fp_list = [v - fp_offset for v in fp_list]  # Remove potential offset from "no change" ("0") results
                fn_list = [v - fn_offset for v in fn_list]  # Remove potential offset from "no change" ("0") results

            fp_val = np.average(fp_list) if isinstance(fp_list, list) and fp_list else 0
            fn_val = np.average(fn_list) if isinstance(fn_list, list) and fn_list else 0
            series[c][fp_string].append(fp_val)
            series[c][fn_string].append(fn_val)

    # Plot
    plt.figure(figsize=(12, 7))
    for c in coordinators:
        # Solid for fp, dashed for fn, same color cycle by plotting and grabbing the color
        # First plot fp and capture color from the created line
        fp_line, = plt.plot(x_vals, series[c][fp_string], label=f"{c} {fp_string}")
        # Use the same color for fn with dashed linestyle
        plt.plot(x_vals, series[c][fn_string], linestyle="--", label=f"{c} {fn_string}", color=fp_line.get_color(), alpha=0.7)
        # Use the same color for fn with dashed linestyle
        plt.plot(x_vals, series['unattributed'][fp_string], linestyle="--", label=f"unattributed", color='gray', alpha=0.5, linewidth=3)

    plt.xlabel("Removed attributions (%)")
    plt.ylabel("Count (#) or ratio (%)")
    plt.title(f"Coordinator '{main_coordinator}' ({num_true_coord_txs} txs) FP (solid) and FN (dashed) over % of removed attributions")
    plt.grid(True, which="both", linestyle=":", linewidth=0.5)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(target_path, filename), dpi=200, bbox_inches="tight")
    plt.close()

    return series, x_vals


def plot_coord_attribution_stats_aggregated(target_path: Path | str, filename: str, label_str: str, omitt_coords: list,
                                            log_scale: bool, join_coord_results: bool=False):
    file_path = os.path.join(target_path, f'{filename}.json')
    results = als.load_json_from_file(file_path)
    fp_string = 'fp'
    fn_string = 'fn'

    assert len(results) == 1, f'Too many top-level keys detected'  # Expect only 1 threshold key
    for threshold in results:
        series_aggregated = {coord: {} for coord in results[threshold]}

        # Option 1: drop random single / drop tail single  results[threshold][coord] contains different results to use
        for coord in results[threshold]:
            series, x_vals = plot_coord_attribution_stats(coord, 0, results[threshold][coord], target_path,
                                                                fp_string, fn_string,
                                                                f"{coord}_coord_discovery_analysis_nominal.png")
            series_aggregated[coord] = {
                # fp_string - all but 'unattributed' coordinator
                fp_string: [sum(vals) for vals in zip(*(series[c][fp_string] for c in series
                                                        if c != 'unattributed'))],
                # Only this coordinator for FN
                fn_string: series[coord][fn_string] if join_coord_results else [-x for x in series[coord][fn_string]],
                'unattributed': series['unattributed'][fp_string],
            }


        if join_coord_results:
            # Option 2: drop random any, series_aggregated[coord] contains same type of results as others
            # (random selection of any coordinator's tx to drop) => aggregate results for even less noise
            if 'unattributed' in series_aggregated:
                num_coords = len(series_aggregated) - 1
            else:
                num_coords = len(series_aggregated)

            series_aggregated_all = {'aggregated': {
                    fp_string: [sum(vals)/num_coords for vals in zip(*(series_aggregated[c][fp_string]
                                                                       for c in series_aggregated
                                                                       if c != 'unattributed'))],
                    fn_string: [sum(vals)/num_coords for vals in zip(*(series_aggregated[c][fn_string]
                                                                       for c in series_aggregated))],
                    'unattributed': [int(sum(vals)/num_coords) for vals in zip(*(series_aggregated[c]['unattributed']
                                                                            for c in series_aggregated))],
                }
            }
            series_aggregated = series_aggregated_all

        def plot_symlog(x_vals: list, series_aggregated: dict, join_coord_results: bool, omitt_coords: list):
            plt.figure(figsize=(10, 5))
            plt.rcParams.update({'font.size': 14})  # Set global font size
            if join_coord_results:
                fp_line, = plt.plot(x_vals, series_aggregated['aggregated'][fp_string], linestyle="-.",
                                    label=f"false positives (all)", alpha=0.7, linewidth=2)
                plt.plot(x_vals, series_aggregated['aggregated'][fn_string], linestyle="--",
                         label=f"false negatives (all)", alpha=0.7, linewidth=2)
                plt.plot(x_vals, series_aggregated['aggregated']['unattributed'], linestyle='-',
                         label=f"unattributed (all)",
                         color='gray', alpha=0.5,
                         linewidth=5)
            else:
                plt.plot([0], [0], linestyle='-.', color='gray', label=f'(false positive)', alpha=0.7, linewidth=2)
                plt.plot([0], [0], linestyle='--', color='gray', label=f'(false negative)', alpha=0.7, linewidth=2)
                for coord in sorted(series_aggregated.keys()):
                    if coord not in omitt_coords:
                        fp_line, = plt.plot(x_vals, series_aggregated[coord][fp_string], linestyle="-.",
                                            label='_nolegend_', alpha=0.7, linewidth=2)
                        plt.plot(x_vals, series_aggregated[coord][fn_string], linestyle="--", label='_nolegend_',
                                 color=fp_line.get_color(),
                                 alpha=0.7, linewidth=2)
                        # plt.plot(x_vals, series_aggregated[coord]['unattributed'], linestyle='-', label=f"{coord} (unattributed)",
                        #          color=fp_line.get_color(), alpha=0.5,
                        #          linewidth=3)

                        # Fake plot to fill legend
                        plt.plot([0], [0], linestyle='-', color=fp_line.get_color(), label=f'{coord}')

            plt.xlabel("Removed attributions (%)")
            plt.ylabel(f"Number of transactions {'(log scale)' if log_scale else ''}")
            plt.title(f"Misattributed transactions after removed attributions ({label_str})")
            plt.grid(True, which="both", linestyle=":", linewidth=0.5)
            plt.legend(handlelength=2, handletextpad=0.8, ncol=2, fontsize=10)

            if log_scale:
                plt.yscale('log' if join_coord_results else 'symlog')

                # Symmetric log
                def symlog_mag_formatter(y, pos):
                    if y == 0:
                        return "0"
                    # show |y| and an arrow for direction
                    magnitude = abs(y)
                    # Compact formatting for large/small values: switch to scientific if needed
                    if magnitude >= 100000 or magnitude < 1e-2:
                        text = f"{magnitude:.0e}"
                    else:
                        # trim trailing zeros
                        text = f"{magnitude:g}"
                    # return f"{text} ↑" if y > 0 else f"{text} ↓"
                    if magnitude == 1:
                        return ""
                    else:
                        return f"{text}" if y > 0 else f"{text}"

                plt.gca().yaxis.set_major_formatter(FuncFormatter(symlog_mag_formatter))

            plt.tight_layout()
            plt.savefig(Path(target_path, f'{filename}_aggregated.png'), dpi=200, bbox_inches="tight")
            plt.close()


        def plot_singleaxis(x_vals: list, series_aggregated: dict, join_coord_results: bool, omitt_coords: list):
            plt.figure(figsize=(10, 5))
            plt.rcParams.update({'font.size': 14})  # Set global font size
            if join_coord_results:
                fp_line, = plt.plot(x_vals, series_aggregated['aggregated'][fp_string], linestyle="-.",
                                    label=f"false positives (all)", alpha=0.7, linewidth=3)
                plt.plot(x_vals, series_aggregated['aggregated'][fn_string], linestyle="--",
                                    label=f"false negatives (all)", alpha=0.7, linewidth=3)
                plt.plot(x_vals, series_aggregated['aggregated']['unattributed'], linestyle='-',
                         label=f"unattributed (all)",
                         color='gray', alpha=0.5,
                         linewidth=5)
            else:
                plt.plot([0], [0], linestyle='-.', color='gray', label=f'(false positive)', alpha=0.7, linewidth=2)
                plt.plot([0], [0], linestyle='--', color='gray', label=f'(false negative)', alpha=0.7, linewidth=2)
                for coord in sorted(series_aggregated.keys()):
                    if coord not in omitt_coords:
                        fp_line, = plt.plot(x_vals, series_aggregated[coord][fp_string], linestyle="-.",
                                            label='_nolegend_', alpha=0.7, linewidth=2)
                        # fp_line, = plt.plot(x_vals, series_aggregated[coord][fp_string], linestyle="-.",
                        #                     label=f"{coord} (misattributed)", alpha=1, linewidth=3)
                        # fp_line, = plt.plot(x_vals, series_aggregated[coord][fp_string], linestyle="-.",
                        #                     label=f"{coord} (false positives)", alpha=0.7, linewidth=3)
                        plt.plot(x_vals, series_aggregated[coord][fn_string], linestyle="--", label='_nolegend_',
                                 color=fp_line.get_color(),
                                 alpha=0.7, linewidth=2)
                        # plt.plot(x_vals, series_aggregated[coord]['unattributed'], linestyle='-', label=f"{coord} (unattributed)",
                        #          color=fp_line.get_color(), alpha=0.5,
                        #          linewidth=3)

                        # Fake plot to fill legend
                        plt.plot([0], [0], linestyle='-', color=fp_line.get_color(), label=f'{coord}')

            plt.xlabel("Removed attributions (%)")
            plt.ylabel(f"Number of transactions {'(log scale)' if log_scale else ''}")
            plt.title(f"Misattributed transactions after removed attributions ({label_str})")
            plt.grid(True, which="both", linestyle=":", linewidth=0.5)
            plt.legend(handlelength=4, handletextpad=0.8, ncol=1)
            if log_scale:
                plt.yscale('log' if join_coord_results else 'symlog')

                ymax = plt.gca().get_ylim()[1]
                max_pow = int(np.floor(np.log10(ymax)))
                major_ticks = [10 ** k for k in range(1, max_pow + 1)]  # 10, 100, ...

                plt.gca().yaxis.set_major_locator(FixedLocator(major_ticks))
                plt.gca().yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
                plt.gca().yaxis.set_minor_formatter(NullFormatter())

            plt.gca().yaxis.set_major_formatter(ScalarFormatter())
            plt.ticklabel_format(style='plain', axis='y')
            plt.tight_layout()
            plt.savefig(Path(target_path, f'{filename}_aggregated.png'), dpi=200, bbox_inches="tight")
            plt.close()

        if join_coord_results:
            plot_singleaxis(x_vals, series_aggregated, join_coord_results, omitt_coords)
        else:
            plot_symlog(x_vals, series_aggregated, join_coord_results, omitt_coords)


def plot_mapping_datasets_stats(cjtxs: dict, mappings: dict, dataset_names: list, target_path: str | Path):
    # Plot number of transactions per day from different datasets
    crawl_coord_txs = {txid: None for dataset, txs in mappings.items() if dataset in dataset_names for txid in txs}

    datasets_dates = {}
    for dataset_name in dataset_names:
        datasets_dates[dataset_name] = {txid: cjtxs['coinjoins'][txid]['broadcast_time']
                                        for txid in mappings[dataset_name].keys()}
    datasets_dates['all'] = {txid: cjtxs['coinjoins'][txid]['broadcast_time']
                                    for txid in cjtxs['coinjoins'].keys()}
    datasets_dates['unattributed'] = {txid: cjtxs['coinjoins'][txid]['broadcast_time']
                                    for txid in cjtxs['coinjoins'].keys() if txid not in crawl_coord_txs}

    als.save_json_to_file_pretty(os.path.join(target_path, 'coordinator_attribution_dataset.json'), datasets_dates)

    # --- aggregate to daily counts per dataset ---
    def daily_counts_from_datasets(ds_dates, tz_localize=None, tz_convert=None) -> pd.DataFrame:
        series_dict = {}
        for name, d in ds_dates.items():
            idx = pd.to_datetime(list(d.values()))
            if tz_localize:
                idx = pd.DatetimeIndex(idx).tz_localize(tz_localize)
            if tz_convert:
                idx = pd.DatetimeIndex(idx).tz_convert(tz_convert)

            s = pd.Series(1, index=idx).sort_index()
            s = s.groupby(s.index.normalize()).size()
            series_dict[name] = s

        all_dates = pd.date_range(
            start=min(s.index.min() for s in series_dict.values()),
            end=max(s.index.max() for s in series_dict.values()),
            freq="D",
        )

        df = pd.DataFrame(index=all_dates)
        for name, s in series_dict.items():
            df[name] = s.reindex(all_dates, fill_value=0).astype(int)
        return df

    # If your timestamps are UTC but you want Europe/Prague day boundaries:
    # df = daily_counts_from_datasets(datasets_dates, tz_localize="UTC", tz_convert="Europe/Prague")
    df = daily_counts_from_datasets(datasets_dates)

    # --- plot (matplotlib) ---
    plt.figure(figsize=(10, 5))
    for col in df.columns:
        if col == 'unattributed':
            plt.fill_between(df.index, df[col], alpha=0.4, label=col, color='red')
            plt.plot(df.index, df[col], linewidth=1.0, alpha=0.9, label="_nolegend_", color='red')
        elif col == 'all':
            plt.plot(df.index, df[col], label=f'all: dumplings + crawl_*', alpha=0.3, linewidth=3, color='gray')
        else:
            plt.fill_between(df.index, df[col], alpha=0.1, label="_nolegend_")
            plt.plot(df.index, df[col], label=f'dataset: {col}', alpha=0.9, linestyle="-.")

    plt.title("Daily coinjoin transactions attributed by different ground-truth datasets (post-zkSNACKs)")
    #plt.xlabel("Date")
    plt.ylabel("Coinjoins per day")
    plt.legend(loc="upper right", fontsize=14)
    plt.grid(True, linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(target_path, 'crawl_datasets.png'), dpi=200, bbox_inches="tight")
    print(f'Saving {target_path}')
    plt.close()