import os
import shutil
import sys
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from orjson import orjson

import cj_process.cj_analysis as als
SATS_IN_BTC = 100000000


def build_plot_intercoord_flows_sankey(base_path: str, entity_dict: dict, transaction_outputs: dict, counts: bool, start_date: str = None, filter_out: set = None):
    results_file = build_intercoord_flows_sankey(base_path, entity_dict, transaction_outputs, counts, start_date, filter_out)
    plot_intercoord_flows_sankey(base_path, results_file, counts, start_date)


def build_intercoord_flows_sankey(base_path: str, entity_dict: dict, transaction_outputs: dict, counts: bool, start_date: str = None, filter_out: set = None):
    if filter_out is None:
        filter_out = set()

    output_file_template = f"coordinator_flows_{'counts' if counts else 'values'}_{start_date[0:10] if start_date else ''}"
    entity_names = list(entity_dict.keys())
    entity_index = {name: i for i, name in enumerate(sorted(entity_names))}
    print(f'build_intercoord_flows_sankey: {entity_index}')

    # Precompute transaction-to-entity mapping for faster lookup
    tx_to_entity = {tx_id: entity for entity, tx_ids in entity_dict.items() for tx_id in tx_ids}

    flows_all = {}  # All flows including to same coordinator
    flows_only_inter = {}  # Only flows to another coordinator
    flows_only_inter_btc = {} # Only flows to another coordinator denominated in sats

    for entity, tx_ids in entity_dict.items():
        print(f'Processing {entity} coordinator', end="")
        for tx_id in tx_ids:
            coinjoin_data = transaction_outputs['coinjoins'].get(tx_id)
            if coinjoin_data:
                # Check if start date parameter was filled in and if yes, check for start date
                if start_date is not None and start_date > transaction_outputs['coinjoins'][tx_id]['broadcast_time']:
                    continue  # Coinjoin happen before start date limit, skip it

                # Process this coinjoin
                for output, output_data in coinjoin_data['outputs'].items():
                    next_tx = output_data.get("spend_by_tx")
                    if next_tx:
                        txid, _ = als.extract_txid_from_inout_string(next_tx)
                        next_entity = tx_to_entity.get(txid)
                        if next_entity:
                            key = (entity, next_entity)
                            if counts:
                                # Count number of outputs
                                flows_all[key] = flows_all.get(key, 0) + 1
                                if entity != next_entity:
                                    flows_only_inter[key] = flows_only_inter.get(key, 0) + 1
                            else:
                                # Count value of outputs
                                flows_all[key] = flows_all.get(key, 0) + output_data.get("value", 0)
                                if entity != next_entity:
                                    flows_only_inter[key] = flows_only_inter.get(key, 0) + output_data.get("value", 0)
        print('... done')

    # Turn from tuples to dict for saving
    flows_to_save = {}
    for (k1, k2), v in flows_only_inter.items():
        flows_to_save.setdefault(k1, {})[k2] = v

    # If required, filter all references specified in filter_out
    for coord, items in list(flows_to_save.items()):
        if coord in filter_out:
            del flows_to_save[coord]
        else:
            for item in list(items):
                if item in filter_out:
                    del items[item]

    if len(filter_out) > 0:
        out_file_name = f"{output_file_template}_filtered_{len(filter_out)}.json"
    else:
        out_file_name = f"{output_file_template}.json"
    save_path = os.path.join(base_path, out_file_name)
    als.save_json_to_file_pretty(save_path, flows_to_save, True)
    print(f"Flows saved to {save_path}")

    return out_file_name


def plot_intercoord_flows_sankey(base_path: str, out_file_name: str, counts: bool, start_date: str = None):
    #output_file_template = f"coordinator_flows_{'counts' if counts else 'values'}_{start_date[0:10] if start_date else ''}"
    output_file_template = Path(out_file_name).with_suffix("")

    # Load from file, turn back from dict to tuples (this )
    flows_loaded = als.load_json_from_file(os.path.join(base_path, out_file_name))
    # Compute required entity structures
    all_names = set(flows_loaded.keys())
    all_names.update([name for x in flows_loaded.keys() for name in sorted(flows_loaded[x].keys())])
    entity_names = list(all_names)
    entity_index = {name: i for i, name in enumerate(entity_names)}
    print(entity_index)
    print(f'plot_intercoord_flows_sankey: {entity_index}')

    flows_only_inter = {(k1, k2): v for k1, inner in flows_loaded.items() for k2, v in inner.items()}


    # Pre-process flows before display
    if not counts:
        flows_only_inter_btc = {key: round(flows_only_inter[key] / SATS_IN_BTC, 1) for key in flows_only_inter.keys()}
        flows = flows_only_inter_btc
    else:
        #flows = flows_all
        flows = flows_only_inter

    type_str = 'counts' if counts else 'values'
    print(f"Inter-coordinators flows ({type_str}): {flows}")

    sources, targets, values = zip(
        *[(entity_index[src], entity_index[tgt], val) for (src, tgt), val in flows.items()]) if flows else ([], [], [])

    def grayscale_color_shades(n, alpha=0.8):
        grays = np.linspace(50, 200, n).astype(int)  # Avoid extremes like pure black/white
        return [f"rgba({g},{g},{g},{alpha})" for g in grays]

    link_colors = grayscale_color_shades(len(values)) if values else []
    # def random_color():
    #     return f"rgba({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)}, 0.8)"
    # link_colors = [random_color() for _ in values]

    # Create the Sankey diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=entity_names
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        )
    ))
    fig.update_layout(
        #font=dict(size=18)  # This affects all text in the Sankey diagram
        font = dict(size=28)  # This affects all text in the Sankey diagram
    )
    fig.update_layout(title_text=f"Inter-coordinators flows for Wasabi 2.x ({'output counts' if counts else 'output values'})"
                                 f" [{'all coinjoins' if start_date is None else 'coinjoins after ' + start_date}]", font_size=18)
    print(f"Sankey diagram updated")
    #fig.show()  # BUGBUG: this call hangs # This ensures the renderer is initialized before saving
    fig.write_html(f"{os.path.join(base_path, f'{output_file_template}.html')}", auto_open=False)
    print(f"Sankey diagram shown")
    # fig.to_html(os.path.join(base_path, f'{output_file_template}.html'))
    # print(f"Sankey diagram to html saved")
    #fig.write_image(os.path.join(base_path, f'{output_file_template}.png'))  # BUGBUG: this call hangs
    print(f"Sankey diagram saved as {os.path.join(base_path, f'{output_file_template}.html')}")


def visualize_coord_flows(base_path: str):
    with open(os.path.join(base_path, 'Scanner', 'wasabi2_others', 'txid_coord_discovered_renamed.json'), "r") as file:
        entities = orjson.loads(file.read())
    # with open(os.path.join(base_path, "wasabi2_others/txid_coord_t.json"), "r") as file:
    #     entities = orjson.loads(file.read())

    load_path = os.path.join(base_path, 'Scanner', 'wasabi2_others', 'coinjoin_tx_info.json')
    with open(load_path, "r") as file:
        data = orjson.loads(file.read())

    ADD_ZKSNACKS = True
    if ADD_ZKSNACKS:
        load_path = os.path.join(base_path, 'Scanner', "wasabi2_zksnacks", 'coinjoin_tx_info.json')
        with open(load_path, "r") as file:
            data_zksnacks = orjson.loads(file.read())
            data['coinjoins'].update(data_zksnacks['coinjoins'])


    # Split entities per months
    #entities_filtered = {entity: entities[entity] for entity in entities.keys() if not entity.isdigit()}
    # entities_months = {}
    # for entity in entities_filtered:
    #     for cjtx in entities_filtered[entity]:
    #         if cjtx in data['coinjoins']:
    #             year_month = data['coinjoins'][cjtx]["broadcast_time"][0:7]
    #             entity_and_year = f'{entity}_{year_month}'
    #             if entity_and_year not in entities_months:
    #                 entities_months[entity_and_year] = []
    #             entities_months[entity_and_year].append(cjtx)

    # Add zksnacks if required
    if ADD_ZKSNACKS:
        entities['zksnacks'] = list(data_zksnacks['coinjoins'].keys())

    # Use all entities
    #entities_to_process = entities
    # Filter only larger coordinators
    #entities_to_process = {entity: entities_to_process[entity] for entity in entities_to_process.keys() if entity in ["kruw", "mega", "btip", "gingerwallet", "wasabicoordinator", "coinjoin_nl", "opencoordinator", "dragonordnance", "wasabist", "zksnacks"]}
    # Filter only coordinators with known name (only digits are discarded)
    entities_to_process = {entity: entities[entity] for entity in entities.keys() if not entity.isdigit()}

    CONFIG_LIST = [{}, {'unattributed'}]
    # Create filtered results with only selected important coordinators shown
    # All values, but drop all references to 'unattributed'
    for to_filter in CONFIG_LIST:
        build_plot_intercoord_flows_sankey(base_path, entities_to_process, data, True, None, to_filter)
        build_plot_intercoord_flows_sankey(base_path, entities_to_process, data, False, None, to_filter)
        build_plot_intercoord_flows_sankey(base_path, entities_to_process, data, True, "2024-09-01 00:00:00.000", to_filter)
        build_plot_intercoord_flows_sankey(base_path, entities_to_process, data, False, "2024-09-01 00:00:00.000", to_filter)
        build_plot_intercoord_flows_sankey(base_path, entities_to_process, data, True, "2025-01-01 00:00:00.000", to_filter)
        build_plot_intercoord_flows_sankey(base_path, entities_to_process, data, False, "2025-01-01 00:00:00.000", to_filter)




def gant_coordinators_plotly(base_path: str):
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    from math import ceil

    # Perform selective renaming of files for proper visualization
    to_rename = [('wasabi1_zksnacks_freshliquidity_values_norm.json', 'wasabi1_Wasabi 1.x (zksnacks)_freshliquidity_values_norm.json'),
                 ('wasabi2_zksnacks_freshliquidity_values_norm.json', 'wasabi2_Wasabi 2.x (zksnacks)_freshliquidity_values_norm.json')]
    for item in to_rename:
        shutil.copy2(os.path.join(base_path, item[0]), os.path.join(base_path, item[1]))

    base_tasks = [
        #dict(Task="Whirlpool all (Sam.)", Start="2019-04-17", Finish="2024-04-24"),
        dict(Task="Wasabi 1.x (zksnacks)", Start="2018-07-19", Finish="2023-05-19", y_pos=0),
        dict(Task="Wasabi 2.x (zksnacks)", Start="2022-06-18", Finish="2024-06-02", y_pos=1),
        dict(Task="Samourai Whirlpool 5M", Start="2019-04-17", Finish="2024-04-24", y_pos=2),
        dict(Task="Samourai Whirlpool 1M", Start="2019-05-23", Finish="2024-04-24", y_pos=3),
        dict(Task="Samourai Whirlpool 50M", Start="2019-08-02", Finish="2024-04-24", y_pos=4),
        dict(Task="Samourai Whirlpool 100k", Start="2021-03-05", Finish="2024-04-24", y_pos=5),
        dict(Task="Wasabi 2.x (kruw.io)", Start="2024-05-31", Finish=datetime.today().strftime("%Y-%m-%d"), y_pos=3),
        dict(Task="Wasabi 2.x (gingerwallet)", Start="2024-05-21", Finish=datetime.today().strftime("%Y-%m-%d"), y_pos=2),
        #dict(Task="Wasabi 2.x (wasabicoordinator)", Start="2024-06-11", Finish="2024-08-11", y_pos=4),
        dict(Task="Wasabi 2.x (opencoordinator)", Start="2024-07-08", Finish=datetime.today().strftime("%Y-%m-%d"), y_pos=4),
        #dict(Task="Wasabi 2.x (others)", Start="2024-05-03", Finish="2024-11-11", y_pos=5),
        #dict(Task="Wasabi 2.x (small coords.)", Start="2024-05-03", Finish=datetime.today().strftime("%Y-%m-%d"), y_pos=5),
        #        dict(Task="Wasabi 2.x (wasabist)", Start="2024-07-23", Finish="2024-08-03", y_pos=9),
        dict(Task="Ashigaru 2.5M", Start="2025-05-31", Finish=datetime.today().strftime("%Y-%m-%d"), y_pos=5),
        dict(Task="Ashigaru 25M", Start="2025-06-06", Finish=datetime.today().strftime("%Y-%m-%d"), y_pos=6),
    ]

    df = pd.DataFrame(base_tasks)
    df["Start"] = pd.to_datetime(df["Start"])
    df["Finish"] = pd.to_datetime(df["Finish"])
    df["Duration"] = (df["Finish"] - df["Start"]).dt.days
    df["Start_ordinal"] = df["Start"].map(datetime.toordinal)
    df["Finish_ordinal"] = df["Finish"].map(datetime.toordinal)
    df["y_pos"] = df["y_pos"]

    fig = go.Figure()

    bar_height = 0.8

    # --- Gantt bars ---
    for _, row in df.iterrows():
        fig.add_shape(
            type="rect",
            x0=row.Start_ordinal,
            x1=row.Finish_ordinal,
            y0=row.y_pos - bar_height / 2,
            y1=row.y_pos + bar_height / 2,
            fillcolor="white",
            line=dict(color="black", width=1),
            layer="above"
        )

        # Add task label manually as an annotation (optional)
        fig.add_annotation(
            x=(row.Start_ordinal + row.Finish_ordinal) / 2,
            y=row.y_pos,
            text=row.Task,
            showarrow=False,
            font=dict(color="black", size=14, weight="bold"),
            xanchor="center",
            yanchor="middle"
        )

    from matplotlib import colormaps
    # Define colormaps per category
    group_colormaps = {
        "Whirlpool": colormaps.get_cmap("Blues"),
        "Wasabi 1.x": colormaps.get_cmap("Reds"),
        "Wasabi 2.x": colormaps.get_cmap("Greens"),
        "Ashigaru": colormaps.get_cmap("Blues")
    }

    for task_idx, row in enumerate(df.itertuples()):
        duration_days = (row.Finish - row.Start).days
        num_bins = max(2, ceil(duration_days / 30))  # 1 bin/month
        bin_width = duration_days / num_bins

        # --- Determine task group and colormap ---
        if "Whirlpool" in row.Task:
            cmap = group_colormaps["Whirlpool"]
        elif "Ashigaru" in row.Task:
            cmap = group_colormaps["Ashigaru"]
        elif "Wasabi 1.x" in row.Task:
            cmap = group_colormaps["Wasabi 1.x"]
        elif "Wasabi 2.x" in row.Task:
            cmap = group_colormaps["Wasabi 2.x"]
        else:
            cmap = colormaps.get_cmap("Greys")  # default fallback

        # Generate synthetic values
        values = np.clip(np.sin(np.linspace(0, np.pi, num_bins)) + 0.1 * np.random.randn(num_bins), 0, 1)
        print(f'Original length for {row.Task}: {len(values)}')

        coords = [('wasabi2', 'kruw'), ('wasabi2', 'Wasabi 2.x (zksnacks)'), ('wasabi2', 'gingerwallet'),
                  ('wasabi2', 'opencoordinator'), ('wasabi2', 'wasabicoordinator'),
                  ('whirlpool', '100k'), ('whirlpool', '1M'), ('whirlpool', '5M'), ('whirlpool', '50M'),
                  ('wasabi1', 'Wasabi 1.x (zksnacks)'), ('whirlpool_ashigaru', '2_5M'), ('whirlpool_ashigaru', '25M')]
        for coord in coords:
            # Load real values
            if coord[1] in row.Task:
                liquidity = als.load_json_from_file(os.path.join(base_path, f'{coord[0]}_{coord[1]}_freshliquidity_values_norm.json'))['months_liquidity'][1:]
                real_values = np.array([key for key, _ in groupby(liquidity)])
                print(f'Collapsed length: {len(real_values)}')
                if len(real_values) != len(values):
                    print(f'Mismatch between expected and loaded value for {coord[1]}; expected: {len(values)}, loaded {len(real_values)} ')
                    if len(real_values) > len(values):
                        real_values = real_values[:len(values)]
                    else:
                        pad_values = np.full((len(values) - len(real_values)), real_values[-1])
                        real_values = np.concatenate([real_values, pad_values])

                values = real_values
        values = values / values.max()

        for i in range(num_bins):
            start_day = row.Start + timedelta(days=i * bin_width)
            end_day = start_day + timedelta(days=bin_width)

            # Convert matplotlib color to Plotly-compatible rgba string
            r, g, b, a = [int(255 * c) if j < 3 else round(c, 2) for j, c in enumerate(cmap(values[i]))]
            color = f"rgba({r}, {g}, {b}, {a})"

            fig.add_shape(
                type="rect",
                x0=start_day.toordinal(),
                x1=end_day.toordinal(),
                y0=row.y_pos - 0.4,
                y1=row.y_pos + 0.4,
                fillcolor=color,
                line=dict(width=0),
                layer="above"
            )

    # --- Axes formatting ---
    # Ensure full vertical range covers all shape bars
    min_y = df["y_pos"].min() - bar_height
    max_y = df["y_pos"].max() + bar_height

    fig.update_yaxes(
        range=[max_y, min_y],  # reversed axis!
        tickvals=[],
        ticktext=[],
        showgrid=False,
        title=None
    )

    # --- Monthly ticks for better performance ---
    start_min = df["Start"].min()
    end_max = df["Finish"].max()
    tick_dates = pd.date_range(start=start_min, end=end_max, freq="MS")  # 1st of each month
    tick_vals = [d.toordinal() for d in tick_dates]

    # Show tick labels only for January, formatted as year
    tick_dates = pd.date_range(start=start_min, end=end_max, freq="MS")  # 1st of each month
    tick_vals = [d.toordinal() for d in tick_dates]
    tick_labels = [d.strftime("%Y") if d.month == 1 else "" for d in tick_dates]

    # Tighten x-axis range to just beyond min/max dates
    start_min = df["Start"].min()
    end_max = df["Finish"].max()

    x_range = [
        (start_min - pd.Timedelta(days=30)).toordinal(),  # 30 days before
        (end_max + pd.Timedelta(days=30)).toordinal(),  # 30 days after
    ]

    fig.update_xaxes(
        range=x_range,
        tickvals=tick_vals,
        ticktext=tick_labels,
        showgrid=True,
        gridcolor='lightgray',
        title=None,
        tickfont=dict(color="black", size=18, family="Arial Bold")
    )
    # --- Adaptive height to fit on one screen ---
    max_chart_height = 600
    bar_padding = 8
    fig_height = min(max_chart_height, 200 + len(df) * bar_padding)

    fig.update_layout(
        #title="Minimalist Gantt Chart with Trend Lines (Monthly Ticks)",
        template='simple_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=40, r=40, t=40, b=40),
        height=fig_height
    )

    # --- Output ---
    fig.write_html("gantt_monthly_ticks_trend_lines.html", auto_open=True)
    fig.write_image("gantt_monthly_ticks_trend_lines.pdf")
    fig.write_image("gantt_monthly_ticks_trend_lines.svg")
    #plt.savefig(f'gantt_monthly_ticks_trend_lines.pdf', dpi=300)


def compute_reorder_stats(base_path: str):
    reorder_stats = als.load_json_from_file(os.path.join(base_path, 'Scanner', 'tx_reordering_stats.json'))

    # Group into 10-minute bins
    grouped = defaultdict(int)
    for minute, value in reorder_stats.items():
        bin_start = (int(minute) // 10) * 10
        grouped[bin_start] += value

    grouped.pop(0, None)
    sorted_items = sorted(grouped.items())
    keys, values = zip(*sorted_items)

    # Plot
    plt.bar(keys, values)
    plt.xlabel('Reordering')
    plt.ylabel('Frequency')
    #plt.title('Bar Plot Sorted by Keys')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        print('No target path provided')
        exit(1)

    print(f'Processing path {base_path}')
    #compute_reorder_stats(base_path)
    #gant_coordinators_plotly(base_path)
    visualize_coord_flows(base_path)


