import csv


def read_csv(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def avg_strength(data):
    total = 0
    count = 0
    for row in data:
        total += float(row["强度"])
        count += 1
    return total / count if count else 0


def count_by_age(data):
    counts = {}
    for row in data:
        age = row["龄期"]
        counts[age] = counts.get(age, 0) + 1
    return counts

if __name__ == "__main__":
    data = read_csv('concrete.csv')
    print("总行数:", len(data))
    print("平均强度", avg_strength(data))
    print("各龄期数量:", count_by_age(data))
