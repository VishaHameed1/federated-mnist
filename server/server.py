class Server:

    def __init__(self, model):

        self.global_model = model

    def update_model(
        self,
        global_weights
    ):

        self.global_model.load_state_dict(
            global_weights
        )