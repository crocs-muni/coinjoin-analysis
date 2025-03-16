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
        print(f'Processing {entity} entity', end="")
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
    fig.show()  # This ensures the renderer is initialized before saving
    fig.to_html(os.path.join(base_path, f'{output_file_template}.html'))
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


if __name__ == "__main__":
    base_path = 'c:/!blockchains/CoinJoin/Dumplings_Stats_20250302/'
    visualize_coord_flows(base_path)
