from alg.fedavg import fedavg


class base(fedavg):
    def __init__(self, args):
        super(base, self).__init__(args)

    def server_aggre(self):
        pass
