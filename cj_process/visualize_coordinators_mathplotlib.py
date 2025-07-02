import matplotlib.pyplot as plt
import matplotlib.sankey as sankey


def build_sankey(entity_dict, transaction_outputs):
    entity_names = list(entity_dict.keys())
    entity_index = {name: i for i, name in enumerate(entity_names)}

    flows = {}

    for entity, tx_ids in entity_dict.items():
        for tx_id in tx_ids:
            if tx_id in transaction_outputs:
                for output in transaction_outputs[tx_id]:
                    next_tx = output.get("next_tx")
                    for next_entity, next_tx_ids in entity_dict.items():
                        if next_tx in next_tx_ids:
                            key = (entity, next_entity)
                            flows[key] = flows.get(key, 0) + output.get("value", 0)

    sources = []
    targets = []
    values = []

    for (src, tgt), val in flows.items():
        sources.append(entity_index[src])
        targets.append(entity_index[tgt])
        values.append(val)

    sankey_diagram = sankey.Sankey()
    for i, (src, tgt) in enumerate(zip(sources, targets)):
        sankey_diagram.add(flows=[-values[i], values[i]], labels=[entity_names[src], entity_names[tgt]],
                           orientations=[0, 0])

    fig, ax = plt.subplots()
    sankey_diagram.finish()
    plt.title("Bitcoin Transaction Flow")
    plt.show()


# Example usage
dummy_entities = {
    "Entity A": ["tx1", "tx2"],
    "Entity B": ["tx3"],
    "Entity C": ["tx4"]
}

dummy_outputs = {
    "tx1": [{"next_tx": "tx3", "value": 0.5}],
    "tx2": [{"next_tx": "tx4", "value": 1.2}]
}

build_sankey(dummy_entities, dummy_outputs)
