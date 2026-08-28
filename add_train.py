import itertools

def append_corpus_slice(
    train_path: str = "train.jsonl",
    corpus_path: str = "full_4m_corpus.jsonl",
    amount: int = 1000000
):
    # Step 1: Count total lines L in train.jsonl
    print("Counting lines in train.jsonl...")
    L = 0
    with open(train_path, "r", encoding="utf-8", newline=None) as f:
        for _ in f:
            L += 1
    print(f"Current train line count (L): {L}")

    # Step 2: Stream slice [L, L + amount] and append line by line
    print(f"Appending lines {L} to {L + amount} from {corpus_path}...")
    appended_count = 0
    with open(corpus_path, "r", encoding="utf-8", newline=None) as f_in, \
         open(train_path, "a", encoding="utf-8", newline="\n") as f_out:

        lines_to_append = itertools.islice(f_in, L, L + amount)

        for line in lines_to_append:
            clean_line = line.rstrip("\r\n")  # strip ANY trailing newline variant
            if not clean_line:
                continue  # skip empty lines instead of writing a blank record
            f_out.write(clean_line + "\n")     # always exactly one newline
            appended_count += 1

    print(f"Successfully appended {appended_count} lines.")
    print(f"New total line count for {train_path}: {L + appended_count}")

if __name__ == "__main__":
    append_corpus_slice()