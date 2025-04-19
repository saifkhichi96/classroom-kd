import csv
import os


def extract_val_accuracy(csv_file):
    val_accuracy_list = []

    with open(csv_file, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            val_accuracy = row.get("val_accuracy")
            if val_accuracy:
                val_accuracy_list.append(float(val_accuracy))

    return val_accuracy_list


def main():
    directory = "/netscratch/sarode/Thesis/imagenet-code/logs/lightning_logs/version_23"  # Change this to the directory containing your CSV files
    output_file = "/netscratch/sarode/Thesis/imagenet-code/logs/output_val_accuracy_ask_teacher.csv"  # Change this to the desired output file name

    all_val_accuracies = {}

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            val_accuracies = extract_val_accuracy(file_path)
            all_val_accuracies[filename] = val_accuracies

    with open(output_file, "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["Row"] + list(all_val_accuracies.keys()))

        for i in range(max(map(len, all_val_accuracies.values()))):
            row = [i + 1]  # assuming a 1-indexed row number
            for key in all_val_accuracies.keys():
                if i < len(all_val_accuracies[key]):
                    row.append(all_val_accuracies[key][i])
                else:
                    row.append("")
            writer.writerow(row)

    print(f"Val accuracies extracted from CSV files and saved to {output_file}")


if __name__ == "__main__":
    main()
