import torch
import matplotlib.pyplot as plt
from dataset import LaserData
import numpy as np


def predict_next(model, input_sequence, output_size, scaler, device):
    model.eval()
    current_seq = input_sequence.clone().detach().to(device)
    # for lstm, gru, ... we need the input_size (1) dimension
    if current_seq.ndimension() == 2:
        current_seq = current_seq.unsqueeze(0)
    predictions = []

    with torch.no_grad():
        for i in range(output_size):
            pred = model(current_seq)

            predictions.append(pred.item())

            # update the sequence:
            new_val = pred.unsqueeze(1)  # Shape: (1, 1, 1)
            current_seq = torch.cat((current_seq[:, 1:, :], new_val), dim=1)

    predictions_np = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions_np)


def plot_predictions(input_sequence, predicted_numbers, scaler):
    input_sequence_original = scaler.inverse_transform(input_sequence)
    input_range = np.arange(0, len(input_sequence_original))
    prediction_range = np.arange(len(input_sequence_original), len(input_sequence_original) + len(predicted_numbers))

    plt.figure(figsize=(15, 6))

    plt.plot(input_range, input_sequence_original, label='Input', color='royalblue',
             marker='o',
             markersize=3,
             linewidth=1,
             alpha=0.8)

    plt.plot(prediction_range, predicted_numbers, label='Iterative Prediction', color='crimson',
             marker='o',
             markersize=3,
             linewidth=1,
             alpha=0.8)

    plt.title("Laser Data Prediction")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    # plt.show()
    plt.savefig('out/200_prediction.png')



from models.lstm_model import LaserLSTM

def iterative_predicting_next_points(model, dataset, data_loader, num, device):
    scaler = dataset.get_scaler()
    X_val, _ = data_loader.dataset.tensors
    last_sequence = X_val[-1]
    print("prediction of next ", num, " is started...")
    predictions = predict_next(model, last_sequence, num, scaler, device)
    plot_predictions(last_sequence, predictions, scaler)

if __name__ == "__main__":
    model_path = "out/models/LaserCNNLSTM_h32_l3100epochs_40seq_RMSprop_lr7.508468627929689e-05__.pth"
    print("model ", model_path, " is loading...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    dataset = LaserData("data/Xtrain.mat", sequence_length=30)
    _, val_loader = dataset.get_loaders(batch_size=32)
    scaler = dataset.get_scaler()


    num_of_prediction = 200
    model = LaserLSTM(
        input_size=1,
        hidden_size=32,
        num_layers=2,
        output_size=1
    ).to(device)

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    print("model is loaded.")


    X_val, _ = val_loader.dataset.tensors
    last_sequence = X_val[-1]


    print("prediction of next ", num_of_prediction, " is started...")
    predictions = predict_next(model, last_sequence, num_of_prediction, scaler, device)
    plot_predictions(last_sequence, predictions, scaler)
    # print(predictions_np)
