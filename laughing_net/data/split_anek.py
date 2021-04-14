from sklearn.model_selection import train_test_split

from src.context import ctx
from src.config import params

def split_data(lines):
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    lines = [line.replace("<|startoftext|>", "") for line in lines]
    lines = [line + "\n" for line in lines]
    lines_train, lines_test = train_test_split(
        lines, 
        test_size=params.data.anek.split.test_size,
        random_state=params.data.anek.split.random_state,
    )
    return lines_train, lines_test

if __name__ == "__main__":
    with open(ctx.data_dir / "raw" / "anek.txt") as f:
        lines = f.readlines()
    lines_train, lines_test = split_data(lines)
    with open(ctx.data_dir / "processed" / "anek_train.txt", "w") as f:
        f.writelines(lines_train)
    with open(ctx.data_dir / "processed" / "anek_test.txt", "w") as f:
        f.writelines(lines_test)
