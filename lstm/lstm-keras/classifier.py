
import csv
def read_csv(filename):
    text_score = dict()
    with open(filename, 'r') as f:
        data = csv.reader(f)
        next(data)
        for row in data:
            text_score[row[0], row[1]] = row[2] 
    return text_score

print(read_csv('text_scores.csv'))
