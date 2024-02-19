import pandas as pd
import json


SYSTEM_CONTENT = "Given a password string, decompose the password string and provide the following data in a JSON dict: 'chunks', 'words', 'structure', 'tags', 'transformations' (if applicable)"


class Pass:
    """A special class to manage the data strucutre of an individual password.
    We expect a labelled password to be decomposed the variables as indicated in
    __init__
    """
    def __init__(
        self,
        password=None,
        chunks=None,
        words=None,
        structure=None,
        tags=None,
        transformations=None
    ):
        self.password = password
        self.chunks = chunks
        self.words = words
        self.structure = structure
        self.tags = tags
        self.transformations = transformations

    def populate_from_df(self, row):
        """Get a Pandas DataFrame row and populate the variables
        from the row given from the DataFrame.
        """
        # Check for transformations
        tx = ''
        if not pd.isna(row.transformations):
            tx = row.transformations

        self.password = row['pwd']
        self.chunks = row['chunks']
        self.words = row['words']
        self.structure = row['structure']
        self.tags = row['tags']
        self.transformations = tx

    def convert_to_json(self):
        json_content = {
            'chunks': self.chunks,
            'words': self.words,
            'structure': self.structure,
            'tags': self.tags,
            'transformations': self.transformations
        }
        return json.dumps(json_content)


    def __repr__(self):
        return f'{self.password}## {self.chunks}## {self.words}## {self.structure}## {self.tags}## {self.transformations}'


def gpt_prompt_train(system_content, user_content, assistant_content) -> dict:
    """Create and return a prompt formatted for OpenAI's API"""
    return {
        "messages": [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]
    }


def create_gpt_line(password: Pass) -> str:
    """Create a jsonl line for GPT to train on"""
    msg = gpt_prompt_train(
        system_content=SYSTEM_CONTENT,
        user_content=password.password,
        assistant_content=password.convert_to_json()
    )
    return f"{json.dumps(msg)}\n"


def main(filename: str, output_filename: str):
    df = pd.read_csv(filename, sep='\t')

    with open(output_filename, 'w') as f:
        for _, row in df.iterrows():
            p = Pass()
            p.populate_from_df(row)

            json_line = create_gpt_line(p)
            f.write(json_line)


if __name__ == '__main__':
    import argparse

    # Read filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--output_filename', type=str, required=True)
    args = parser.parse_args()

    # Run the conversion
    main(args.filename, args.output_filename)


