from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import RMSprop, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from dataset import LaserData
from evaluate import evaluate_model
from models.cnn_gru_model import LaserCNNGRU
from models.cnn_lstm_model import LaserCNNLSTM
import torch

from models.cnn_model import LaserCNN
from models.gru_model import LaserGRU
from models.lstm_model import LaserLSTM
from prediction import iterative_predicting_next_points
from train import train_model
from util import set_seed


def reset_weights(m):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()


def get_data_loaders(seq):
    set_seed(100)
    dataset_80 = LaserData("data/Xtrain.mat", split_ratio=0.8, sequence_length=seq)
    scaler_80 = dataset_80.get_scaler()
    train_loader_80, val_loader_80 = dataset_80.get_loaders()
    dataset_test_80 = LaserData(
        "data/Xtest.mat",
        split_ratio=0.8,
        sequence_length=seq,
        scaler=scaler_80,
        mode="test",
        key="Xtest",
    )
    test_loader_80 = dataset_test_80.get_loaders()
    set_seed(100)
    dataset_100 = LaserData("data/Xtrain.mat", split_ratio=1, sequence_length=seq)
    scaler_100 = dataset_100.get_scaler()
    dataset_test_100 = LaserData(
        "data/Xtest.mat",
        split_ratio=1,
        sequence_length=seq,
        scaler=scaler_100,
        mode="test",
        key="Xtest",
    )
    test_loader_100 = dataset_test_100.get_loaders()
    train_loader_100 = dataset_100.get_loaders()
    set_seed(100)
    dataset_90 = LaserData("data/Xtrain.mat", split_ratio=0.9, sequence_length=seq)
    scaler_90 = dataset_90.get_scaler()
    dataset_test_90 = LaserData(
        "data/Xtest.mat",
        split_ratio=0.9,
        sequence_length=seq,
        scaler=scaler_90,
        mode="test",
        key="Xtest",
    )
    test_loader_90 = dataset_test_90.get_loaders()
    train_loader_90, val_loader_90 = dataset_90.get_loaders()

    return (
        test_loader_80,
        test_loader_90,
        test_loader_100,
        train_loader_80,
        train_loader_90,
        train_loader_100,
        val_loader_80,
        val_loader_90,
        dataset_80,
        dataset_90,
        dataset_100,
    )


models = {
    "CNN": {
        "model": "CNN",
        "seq": 50,
        "cnndrop": 0.1,
        "fcdrop": 0,
        "ep": 60,
        "opt": AdamW,
        "scale": StandardScaler,
        "lr": 0.001,
        "wd": 0.0001,
        "kern": 5,
        "filt": 64,
        "schedf": 0.5,
        "schedp": 5,
        "mse": 2.2460,
    },
    "CNNLSTM": {
        "model": "CNNLSTM",
        "seq": 45,
        "cnndrop": 0,
        "lstmdrop": 0,
        "fcdrop": 0,
        "ln": False,
        "ep": 100,
        "opt": RMSprop,
        "scale": StandardScaler,
        "lr": 0.001,
        "wd": 0.0001,
        "hid": 32,
        "layers": 3,
        "kernel": 3,
        "filters": 64,
        "schedf": 0.75,
        "schedp": 5,
        "mse": 1.2185,
    },
    "CNNGRU": {
        "model": "CNNGRU",
        "seq": 40,
        "cnndrop": 0,
        "lstmdrop": 0,
        "fcdrop": 0,
        "ln": False,
        "ep": 100,
        "opt": AdamW,
        "scale": StandardScaler,
        "lr": 0.001,
        "wd": 0.0001,
        "hid": 64,
        "layers": 3,
        "kernel": 5,
        "filters": 64,
        "schedf": 0.75,
        "schedp": 5,
        "mse": 0.9875,
    },
    "GRU": {
        "model": "GRU",
        "seq": 45,
        "lstmdrop": 0,
        "fcdrop": 0.1,
        "ln": False,
        "ep": 30,
        "opt": RMSprop,
        "scale": StandardScaler,
        "lr": 0.001,
        "wd": 0.00001,
        "hid": 64,
        "layers": 3,
        "schedf": 0.75,
        "schedp": 5,
        "mse": 1.8710,
    },
    "LSTM": {
        "model": "LSTM",
        "seq": 25,
        "lstmdrop": 0,
        "fcdrop": 0.1,
        "ln": False,
        "ep": 30,
        "opt": AdamW,
        "scale": StandardScaler,
        "lr": 0.0005,
        "wd": 0.00001,
        "hid": 64,
        "layers": 3,
        "schedf": 0.75,
        "schedp": 5,
        "mse": 2.1654,
    },
}


def initialize_cnnlstm_model(params, device):
    set_seed(100)
    model = LaserCNNLSTM(
        seq_length=params["seq"],
        input_channels=1,
        hidden_size=params["hid"],
        num_layers=params["layers"],
        output_size=1,
        num_filters=params["filters"],
        kernel_size=params["kernel"],
        cnn_dropout=params["cnndrop"],
        lstm_dropout=params["lstmdrop"],
        fc_dropout=params["fcdrop"],
        layer_norm=params["ln"],
    ).to(device)
    return model


def initialize_cnngru_model(params, device):
    set_seed(100)
    model = LaserCNNGRU(
        seq_length=params["seq"],
        input_channels=1,
        hidden_size=params["hid"],
        num_layers=params["layers"],
        output_size=1,
        num_filters=params["filters"],
        kernel_size=params["kernel"],
        cnn_dropout=params["cnndrop"],
        lstm_dropout=params["lstmdrop"],
        fc_dropout=params["fcdrop"],
        layer_norm=params["ln"],
    ).to(device)
    return model


def initialize_cnn_model(params, device):
    set_seed(100)
    model = LaserCNN(
        seq_length=params["seq"],
        num_filters=params["filt"],
        kernel_size=params["kern"],
        dropout=params["cnndrop"],
        fc_dropout=params["fcdrop"],
    ).to(device)
    return model


def initialize_gru_model(params, device):
    set_seed(100)
    model = LaserGRU(
        input_size=1,
        hidden_size=params["hid"],
        num_layers=params["layers"],
        output_size=1,
        dropout=params["lstmdrop"],
        fc_dropout=params["fcdrop"],
        layer_norm=params["ln"],
    ).to(device)
    return model


def initialize_lstm_model(params, device):
    set_seed(100)
    model = LaserLSTM(
        input_size=1,
        hidden_size=params["hid"],
        num_layers=params["layers"],
        output_size=1,
        dropout=params["lstmdrop"],
        fc_dropout=params["fcdrop"],
        layer_norm=params["ln"],
    ).to(device)
    return model


def initialize_optimizer_scheduler(params, mod):
    optimizer = params["opt"](
        mod.parameters(), lr=params["lr"], weight_decay=params["wd"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=params["schedf"],
        patience=params["schedp"],
    )
    return optimizer, scheduler


TEST_CNNLSTM = True
TEST_CNNGRU = True
TEST_LSTM = True
TEST_GRU = True
TEST_CNN = True

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(100)

    CNNLSTM_model_path = "out/models/seed100/CNNLSTM_seq45_cnndrop0_lstmdrop0_fcdrop0_lnFalse_ep100_optRMSprop_scaleStandardScaler_lr0.001_wd0.0001_hid32_layers3_kernel3_filters64_schedf0.75_schedp5_mse1.2185.pth"
    CNNGRU_model_path = "out/models/seed100/CNNGRU_seq40_cnndrop0_lstmdrop0_fcdrop0_lnFalse_ep100_optAdamW_scaleStandardScaler_lr0.001_wd0.0001_hid64_layers3_kernel5_filters64_schedf0.75_schedp5_mse0.9875.pth"
    LSTM_model_path = "out/models/seed100/LSTM_seq25_lstmdrop0_fcdrop0.1_lnFalse_ep30_optAdamW_scaleStandardScaler_lr0.0005_wd1e-05_hid64_layers3_schedf0.75_schedp5_mse2.1654.pth"
    GRU_model_path = "out/models/seed100/GRU_seq45_lstmdrop0_fcdrop0.1_lnFalse_ep30_optRMSprop_scaleStandardScaler_lr0.001_wd1e-05_hid64_layers3_schedf0.75_schedp5_mse1.8710.pth"
    CNN_model_path = "out/models/seed100/CNN_seq50_cnndrop0.1_fcdrop0_ep60_optAdamW_scaleStandardScaler_lr0.001_wd0.0001_kern5_filt64_schedf0.5_schedp5_mse2.2460.pth"

    print("If all models are not evaluated, check the flag")

    if TEST_CNNLSTM:
        # ===================================================CNNLSTM==================================
        data = models["CNNLSTM"]
        (
            test_loader_80,
            test_loader_90,
            test_loader_100,
            train_loader_80,
            train_loader_90,
            train_loader_100,
            val_loader_80,
            val_loader_90,
            dataset_80,
            dataset_90,
            dataset_100,
        ) = get_data_loaders(data["seq"])
        model = initialize_cnnlstm_model(data, device)
        criterion = nn.MSELoss()

        print("=================CNNLSTM=================")
        optimizer, scheduler = initialize_optimizer_scheduler(data, model)
        epochs = data["ep"]

        # train on whole data
        train_model(
            epochs,
            model,
            optimizer,
            criterion,
            train_loader_100,
            None,
            device,
            scheduler=scheduler,
            version="_all_data",
            save_model=True,
        )
        mae, mse = evaluate_model(model, test_loader_100, device, dataset_100)
        print("LSTMCNN results for the best hyperparams trained on 100% data")
        print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")
        iterative_predicting_next_points(
            model, dataset_100, test_loader_100, 200, device
        )

        # train on 90 percet
        model = initialize_cnnlstm_model(data, device)
        optimizer, scheduler = initialize_optimizer_scheduler(data, model)
        train_model(
            epochs,
            model,
            optimizer,
            criterion,
            train_loader_90,
            val_loader_90,
            device,
            scheduler=scheduler,
            version="_90_data",
        )
        mae, mse = evaluate_model(model, test_loader_90, device, dataset_90)
        print("LSTMCNN results for the best hyperparams trained on 90% data")
        print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")

        # train on 80 percent
        model = initialize_cnnlstm_model(data, device)
        optimizer, scheduler = initialize_optimizer_scheduler(data, model)
        train_model(
            epochs,
            model,
            optimizer,
            criterion,
            train_loader_80,
            val_loader_80,
            device,
            scheduler=scheduler,
            version="_80_data",
        )
        mae, mse = evaluate_model(model, test_loader_80, device, dataset_80)

        print("LSTMCNN results for the best hyperparams trained on 80% data")
        print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")

        # evaluate previously trained model
        model = initialize_cnnlstm_model(data, device)
        state_dict = torch.load(CNNLSTM_model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        mae, mse = evaluate_model(model, test_loader_80, device, dataset_80)
        print("LSTMCNN results for the best hyperparams saved model")
        print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")

    if TEST_CNNGRU:
        # ===================================================CNNGRU==================================
        data = models["CNNGRU"]
        (
            test_loader_80,
            test_loader_90,
            test_loader_100,
            train_loader_80,
            train_loader_90,
            train_loader_100,
            val_loader_80,
            val_loader_90,
            dataset_80,
            dataset_90,
            dataset_100,
        ) = get_data_loaders(data["seq"])
        model = initialize_cnngru_model(data, device)
        criterion = nn.MSELoss()

        print("=================CNNGRU=================")
        optimizer, scheduler = initialize_optimizer_scheduler(data, model)
        epochs = data["ep"]

        # train on whole data
        train_model(
            epochs,
            model,
            optimizer,
            criterion,
            train_loader_100,
            None,
            device,
            scheduler=scheduler,
            version="_all_data",
            save_model=True,
        )
        mae, mse = evaluate_model(model, test_loader_100, device, dataset_100)

        print("CNNGRU results for the best hyperparams trained on 100% data")
        print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")
        iterative_predicting_next_points(
            model, dataset_100, test_loader_100, 200, device
        )

        # train on 90 percet
        model = initialize_cnngru_model(data, device)
        optimizer, scheduler = initialize_optimizer_scheduler(data, model)
        train_model(
            epochs,
            model,
            optimizer,
            criterion,
            train_loader_90,
            val_loader_90,
            device,
            scheduler=scheduler,
            version="_90_data",
        )
        mae, mse = evaluate_model(model, test_loader_90, device, dataset_90)
        print("CNNGRU results for the best hyperparams trained on 90% data")
        print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")

        # train on 80 percent
        model = initialize_cnngru_model(data, device)
        optimizer, scheduler = initialize_optimizer_scheduler(data, model)
        train_model(
            epochs,
            model,
            optimizer,
            criterion,
            train_loader_80,
            val_loader_80,
            device,
            scheduler=scheduler,
            version="_80_data",
        )
        mae, mse = evaluate_model(model, test_loader_80, device, dataset_80)

        print("CNNGRU results for the best hyperparams trained on 80% data")
        print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")

        # evaluate previously trained model
        model = initialize_cnngru_model(data, device)
        state_dict = torch.load(CNNGRU_model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        mae, mse = evaluate_model(model, test_loader_80, device, dataset_80)
        print("CNNGRU results for the best hyperparams saved model")
        print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")

    if TEST_CNN:
        # ==================================CNN=====================================

        data = models["CNN"]
        (
            test_loader_80,
            test_loader_90,
            test_loader_100,
            train_loader_80,
            train_loader_90,
            train_loader_100,
            val_loader_80,
            val_loader_90,
            dataset_80,
            dataset_90,
            dataset_100,
        ) = get_data_loaders(data["seq"])
        model = initialize_cnn_model(data, device)

        print("=================CNN=================")
        optimizer, scheduler = initialize_optimizer_scheduler(data, model)
        criterion = nn.MSELoss()

        # train on whole data
        epochs = data["ep"]
        train_model(
            epochs,
            model,
            optimizer,
            criterion,
            train_loader_100,
            None,
            device,
            scheduler=scheduler,
            version="_all_data",
        )

        mae, mse = evaluate_model(model, test_loader_100, device, dataset_100)
        print("CNN results for the best hyperparams trained on 100% data")
        print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")

        # train on 90 percet
        model = initialize_cnn_model(data, device)
        optimizer, scheduler = initialize_optimizer_scheduler(data, model)
        train_model(
            epochs,
            model,
            optimizer,
            criterion,
            train_loader_90,
            val_loader_90,
            device,
            scheduler=scheduler,
            version="_90_data",
        )

        mae, mse = evaluate_model(model, test_loader_90, device, dataset_90)
        print("CNN results for the best hyperparams trained on 90% data")
        print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")

        # train on 80 percet
        model = initialize_cnn_model(data, device)
        optimizer, scheduler = initialize_optimizer_scheduler(data, model)
        train_model(
            epochs,
            model,
            optimizer,
            criterion,
            train_loader_80,
            val_loader_80,
            device,
            scheduler=scheduler,
            version="_80_data",
        )

        mae, mse = evaluate_model(model, test_loader_80, device, dataset_80)
        print("CNN results for the best hyperparams trained on 80% data")
        print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")

        # evaluate previously trained model
        model = initialize_cnn_model(data, device)
        state_dict = torch.load(CNN_model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        mae, mse = evaluate_model(model, test_loader_80, device, dataset_80)
        print("CNN results for the best hyperparams saved model")
        print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")

    if TEST_GRU:
        # ==============================================GRU====================================

        data = models["GRU"]
        (
            test_loader_80,
            test_loader_90,
            test_loader_100,
            train_loader_80,
            train_loader_90,
            train_loader_100,
            val_loader_80,
            val_loader_90,
            dataset_80,
            dataset_90,
            dataset_100,
        ) = get_data_loaders(data["seq"])
        model = initialize_gru_model(data, device)

        print("=================GRU=================")

        # train on whole data
        optimizer, scheduler = initialize_optimizer_scheduler(data, model)
        epochs = data["ep"]
        criterion = nn.MSELoss()
        train_model(
            epochs,
            model,
            optimizer,
            criterion,
            train_loader_100,
            None,
            device,
            scheduler=scheduler,
            version="_all_data",
        )

        mae, mse = evaluate_model(model, test_loader_100, device, dataset_100)
        print("GRU results for the best hyperparams trained on 100% data")
        print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")

        # train on 90 percet
        model = initialize_gru_model(data, device)
        optimizer, scheduler = initialize_optimizer_scheduler(data, model)
        train_model(
            epochs,
            model,
            optimizer,
            criterion,
            train_loader_90,
            val_loader_90,
            device,
            scheduler=scheduler,
            version="_90_data",
        )

        mae, mse = evaluate_model(model, test_loader_90, device, dataset_90)
        print("GRU results for the best hyperparams trained on 90% data")
        print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")

        # train on 80 percet
        model = initialize_gru_model(data, device)
        optimizer, scheduler = initialize_optimizer_scheduler(data, model)
        train_model(
            epochs,
            model,
            optimizer,
            criterion,
            train_loader_80,
            val_loader_80,
            device,
            scheduler=scheduler,
            version="_80_data",
        )

        mae, mse = evaluate_model(model, test_loader_80, device, dataset_80)
        print("GRU results for the best hyperparams trained on 80% data")
        print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")

        # evaluate previously trained model
        model = initialize_gru_model(data, device)
        state_dict = torch.load(GRU_model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        mae, mse = evaluate_model(model, test_loader_80, device, dataset_80)
        print("GRU results for the best hyperparams saved model")
        print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")

    if TEST_LSTM:
        # =============LSTM=============
        data = models["LSTM"]
        (
            test_loader_80,
            test_loader_90,
            test_loader_100,
            train_loader_80,
            train_loader_90,
            train_loader_100,
            val_loader_80,
            val_loader_90,
            dataset_80,
            dataset_90,
            dataset_100,
        ) = get_data_loaders(data["seq"])
        model = initialize_lstm_model(data, device)

        print("=================LSTM=================")

        # train on whole data
        optimizer, scheduler = initialize_optimizer_scheduler(data, model)
        epochs = data["ep"]
        criterion = nn.MSELoss()
        train_model(
            epochs,
            model,
            optimizer,
            criterion,
            train_loader_100,
            None,
            device,
            scheduler=scheduler,
            version="_all_data",
        )

        mae, mse = evaluate_model(model, test_loader_100, device, dataset_100)
        print("LSTM results for the best hyperparams trained on 100% data")
        print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")

        # train on 90 percet
        model = initialize_lstm_model(data, device)
        optimizer, scheduler = initialize_optimizer_scheduler(data, model)
        train_model(
            epochs,
            model,
            optimizer,
            criterion,
            train_loader_90,
            val_loader_90,
            device,
            scheduler=scheduler,
            version="_90_data",
        )

        mae, mse = evaluate_model(model, test_loader_90, device, dataset_90)
        print("LSTM results for the best hyperparams trained on 90% data")
        print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")

        # train on 80 percet
        model = initialize_lstm_model(data, device)
        optimizer, scheduler = initialize_optimizer_scheduler(data, model)
        train_model(
            epochs,
            model,
            optimizer,
            criterion,
            train_loader_80,
            val_loader_80,
            device,
            scheduler=scheduler,
            version="_80_data",
        )

        mae, mse = evaluate_model(model, test_loader_80, device, dataset_80)
        print("LSTM results for the best hyperparams trained on 80% data")
        print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")

        # evaluate previously trained model
        model = initialize_lstm_model(data, device)
        state_dict = torch.load(LSTM_model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        mae, mse = evaluate_model(model, test_loader_80, device, dataset_80)
        print("LSTM results for the best hyperparams saved model")
        print(f"Ori MAE: {mae:.4f}, Ori MSE: {mse:.4f}")
