import pandas as pd

# 1) Read the CSV
df = pd.read_csv('GHI_on_Hottest_Day.csv')

# 2) Make sure it's sorted by time (it should be already)
df = df.sort_values('Datetime').reset_index(drop=True)

# 3) Build the profile: slot 0 = midnight–00:15, slot 1 = 00:15–00:30, …, slot 95 = 23:45–24:00
irr_profile = {i: float(ghi) for i, ghi in enumerate(df['GHI'])}
# print(irr_profile)
