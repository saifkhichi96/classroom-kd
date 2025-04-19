import csv
import matplotlib.pyplot as plt


def plot_val_accuracies(csv_file, output_plot_file):
    rows = []
    with open(csv_file, "r") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    # Extract headers and data
    headers = list(rows[0].keys())
    data = {
        header: [float(row[header]) * 100 if row[header] else None for row in rows[1:]]
        for header in headers
    }
    data["epoch"][:] = [x / 100 for x in data["epoch"]]
    # Plotting
    plt.figure(figsize=(12, 6))
    for header in headers[1:]:  # Skip the 'Row' column
        plt.plot(data["epoch"], data[header], label=header)

    plt.title("Validation Accuracies Trends")
    plt.xlabel("epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(output_plot_file)
    plt.show()


if __name__ == "__main__":
    csv_file_path = (
        "/netscratch/sarode/Thesis/imagenet-code/logs/output_val_accuracy.csv"
    )
    output_plot_file_path = (
        "/netscratch/sarode/Thesis/imagenet-code/logs/val_accuracies_plot_ask.png"
    )
    plot_val_accuracies(csv_file_path, output_plot_file_path)
    print("done")
