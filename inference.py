# local code only
import pandas as pd
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


# code for all
import torch
from src.torch_models import EmbedMLP

model = EmbedMLP(input_dim=301, num_embeddings=3775).cuda()
model.load_state_dict(torch.load("dropout_ep9.pth"))
model.eval()

# local code only
df = pd.read_csv(os.environ["dataset_dir"] + "train_sample.csv", index_col=0)

X = df.drop(["target", "row_id", "time_id"], axis=1).values

for line in tqdm(X):
    values = model(torch.Tensor(line.reshape(1, -1)).cuda())

# TODO: correct test df investment processing
# kaggle code only
# import ubiquant
# env = ubiquant.make_env()
# iter_test = env.iter_test()
# for (test_df, sample_prediction_df) in iter_test:
#     test_df['investment_id'] += 1
#     values = model(torch.Tensor(test_df.drop(['row_id'], axis=1).to_numpy()).cuda())
#     sample_prediction_df['target'] = values.cpu().detach().numpy()  # make your predictions here
#     env.predict(sample_prediction_df)
