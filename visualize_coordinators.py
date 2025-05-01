import os

import plotly.graph_objects as go
from orjson import orjson
import cj_analysis as als


def build_intercoord_flows_sankey_good(base_path: str, entity_dict: dict, transaction_outputs: dict, counts: bool, start_date: str = None):
    output_file_template = f"coordinator_flows_{'counts' if counts else 'values'}"
    entity_names = list(entity_dict.keys())
    entity_index = {name: i for i, name in enumerate(entity_names)}

    # Precompute transaction-to-entity mapping for faster lookup
    tx_to_entity = {tx_id: entity for entity, tx_ids in entity_dict.items() for tx_id in tx_ids}

    flows_all = {}  # All flows including to same coordinator
    flows_only_inter = {}  # Only flows to another coordinator

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

    #flows = flows_all
    flows = flows_only_inter

    print(f'Inter-coordinators flows ({'counts' if counts else 'values'}): {flows}')
    #als.save_json_to_file_pretty(f'{output_file_template}.json', flows)

    sources, targets, values = zip(
        *[(entity_index[src], entity_index[tgt], val) for (src, tgt), val in flows.items()]) if flows else ([], [], [])

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20,
            label=entity_names
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
        )
    ))
    fig.update_layout(title_text=f"Inter-coordinators flows for Wasabi 2.x ({'output counts' if counts else 'output values'})"
                                 f" [{'all coinjoins' if start_date is None else 'coinjoins after ' + start_date}]", font_size=10)
    print(f"Sankey diagram updated")
    #fig.show()  # BUGBUG: this call hangs # This ensures the renderer is initialized before saving
    fig.write_html(f'{output_file_template}.html', auto_open=True)
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

    ADD_ZKSNACKS = False
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

    build_intercoord_flows_sankey_good(base_path, entities_to_process, data, True)
    build_intercoord_flows_sankey_good(base_path, entities_to_process, data, False)
    build_intercoord_flows_sankey_good(base_path, entities_to_process, data, True, "2024-09-01 00:00:00.000")
    build_intercoord_flows_sankey_good(base_path, entities_to_process, data, False, "2024-09-01 00:00:00.000")
    build_intercoord_flows_sankey_good(base_path, entities_to_process, data, True, "2025-01-01 00:00:00.000")
    build_intercoord_flows_sankey_good(base_path, entities_to_process, data, False, "2025-01-01 00:00:00.000")


def gant_coordinators_plotly():
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    from math import ceil

    # --- Define tasks based on your table ---
    base_tasks = [
        #dict(Task="Whirlpool all (Sam.)", Start="2019-04-17", Finish="2024-04-24"),
        dict(Task="Wasabi 1.x (zkSNACKs)", Start="2018-07-19", Finish="2024-06-01", y_pos=0),
        dict(Task="Wasabi 2.x (zkSNACKs)", Start="2022-06-18", Finish="2024-06-01", y_pos=1),
        dict(Task="Whirlpool 5M", Start="2019-04-17", Finish="2024-04-24", y_pos=2),
        dict(Task="Whirlpool 1M", Start="2019-05-23", Finish="2024-04-24", y_pos=3),
        dict(Task="Whirlpool 50M", Start="2019-08-02", Finish="2024-04-24", y_pos=4),
        dict(Task="Whirlpool 100k", Start="2021-03-05", Finish="2024-04-24", y_pos=5),
        dict(Task="Wasabi 2.x (kruw.io)", Start="2024-05-31", Finish=datetime.today().strftime("%Y-%m-%d"), y_pos=6),
        dict(Task="Wasabi 2.x (gingerwallet)", Start="2024-05-31", Finish=datetime.today().strftime("%Y-%m-%d"), y_pos=7),
        dict(Task="Wasabi 2.x (opencoordinator)", Start="2024-05-31", Finish=datetime.today().strftime("%Y-%m-%d"), y_pos=8),
        dict(Task="Wasabi 2.x (wasabist)", Start="2024-05-31", Finish=datetime.today().strftime("%Y-%m-%d"), y_pos=9),
        dict(Task="Wasabi 2.x (wasabicoordinator)", Start="2024-05-31", Finish=datetime.today().strftime("%Y-%m-%d"), y_pos=10),
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
        fig.add_trace(go.Bar(
            y=[row["y_pos"]],
            x=[row["Duration"]],
            base=[row["Start_ordinal"]],
            width=bar_height,
            orientation='h',
            marker=dict(color='white', line=dict(color='black', width=1)),
            text=[row["Task"]],
            textposition='inside',
            insidetextanchor='middle',
            hoverinfo='none',
            showlegend=False
        ))

    # --- Replace trend lines with heatmap fill per task ---
    from matplotlib import cm

    # Define colormaps per category
    group_colormaps = {
        "Whirlpool": cm.get_cmap("Blues"),
        "Wasabi 1.x": cm.get_cmap("Reds"),
        "Wasabi 2.x": cm.get_cmap("Greens")
    }

    for task_idx, row in enumerate(df.itertuples()):
        duration_days = (row.Finish - row.Start).days
        num_bins = max(2, ceil(duration_days / 30))  # 1 bin/month
        bin_width = duration_days / num_bins

        # --- Determine task group and colormap ---
        if "Whirlpool" in row.Task:
            cmap = group_colormaps["Whirlpool"]
        elif "Wasabi 1.x" in row.Task:
            cmap = group_colormaps["Wasabi 1.x"]
        elif "Wasabi 2.x" in row.Task:
            cmap = group_colormaps["Wasabi 2.x"]
        else:
            cmap = cm.get_cmap("Greys")  # default fallback

        # Generate synthetic values
        values = np.clip(np.sin(np.linspace(0, np.pi, num_bins)) + 0.1 * np.random.randn(num_bins), 0, 1)
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
    fig.update_yaxes(
        tickvals=[],
        ticktext=[],
        autorange='reversed',
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
        title=None
    )
    # --- Adaptive height to fit on one screen ---
    max_chart_height = 600
    bar_padding = 8
    fig_height = min(max_chart_height, 200 + len(df) * bar_padding)

    fig.update_layout(
        title="Minimalist Gantt Chart with Trend Lines (Monthly Ticks)",
        template='simple_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=40, r=40, t=40, b=40),
        height=fig_height
    )

    # --- Output ---
    fig.write_html("gantt_monthly_ticks_trend_lines.html", auto_open=True)
    fig.write_image("gantt_monthly_ticks_trend_lines.svg")



if __name__ == "__main__":
    base_path = 'c:/!blockchains/CoinJoin/Dumplings_Stats_20250302/'
    gant_coordinators_plotly()
    #visualize_coord_flows(base_path)
