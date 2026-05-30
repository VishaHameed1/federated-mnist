import copy

def fedavg(client_weights):

    avg_weights = copy.deepcopy(
        client_weights[0]
    )

    for key in avg_weights.keys():

        for i in range(
            1,
            len(client_weights)
        ):

            avg_weights[key] += (
                client_weights[i][key]
            )

        avg_weights[key] = (
            avg_weights[key]
            /
            len(client_weights)
        )

    return avg_weights