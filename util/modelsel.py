from network.models import AlexNet, PamapModel, lenet5v
import copy


def modelsel(args, device):
    if args.dataset in ['vlcs', 'pacs', 'off_home', 'off-cal', 'covid']:
        server_model = AlexNet(num_classes=args.num_classes).to(device)
    elif 'medmnist' in args.dataset:
        server_model = lenet5v().to(device)
    elif 'pamap' in args.dataset:
        server_model = PamapModel().to(device)

    client_weights = [1/args.n_clients for _ in range(args.n_clients)]
    models = [copy.deepcopy(server_model).to(device)
              for _ in range(args.n_clients)]
    return server_model, models, client_weights
