import pandas as pd
import numpy as np

# raw data 불러오기
df = pd.read_csv('~/kaggle-hms/data/raw/train.csv')

# target 칼럼 설정
TARGETS = df.columns[-6:]

# groupby 작업1 - eeg, spectrogram의 id와 offset 최솟값 넣기
train = (
    df
    .groupby(
        ['eeg_id', 'expert_consensus']
        )[
            [
                'eeg_label_offset_seconds', 
                'spectrogram_id',
                'label_id',
                'spectrogram_label_offset_seconds'
            ]
         ]
    .agg(
        {
            'eeg_label_offset_seconds': 'min', 
            'spectrogram_id': 'first',
            'label_id': 'first',
            'spectrogram_label_offset_seconds': 'min'
        })
)
train.columns = ['eeg_min', 'spectogram_id', 'label_id', 'spec_min']

# groupby 작업2 - eeg, spectrogram의 offset 최댓값 넣기
tmp = (
    df
    .groupby(
        ['eeg_id', 'expert_consensus']
        )[['eeg_label_offset_seconds','spectrogram_label_offset_seconds']]
    .agg(
        {
            'eeg_label_offset_seconds': 'max', 
            'spectrogram_label_offset_seconds':'max'
        })
)
train[['eeg_max', 'spec_max']] = tmp

# groupby 작업3 - 환자 id 넣기
tmp = (
    df
    .groupby(['eeg_id', 'expert_consensus'])[['patient_id']]
    .agg('first')
)
train['patient_id'] = tmp

# target 칼럼 확률로 바꿔넣기
tmp = df.groupby(['eeg_id', 'expert_consensus'])[TARGETS].agg('sum')
for t in TARGETS:
    train[t] = tmp[t].values
    
y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1,keepdims=True)
train[TARGETS] = y_data

# index 정리하고 오름차순 정렬하기
train.reset_index(inplace=True)
train.sort_values(by=['eeg_id', 'eeg_min'], inplace=True, ignore_index=True)
train.rename(columns={'expert_consensus': 'target'}, inplace=True)
train = train[
                ['target', 'eeg_id', 'eeg_min', 'eeg_max',
                 'spectogram_id', 'spec_min', 'spec_max', 'label_id', 'patient_id', 
                 'seizure_vote', 'lpd_vote', 'gpd_vote',
                 'lrda_vote', 'grda_vote', 'other_vote']
        ]

# eeg_id 당 병명이 여러 개 나온 경우의 index 추출하기
condition = train['eeg_id'].value_counts()
idx_condition = condition[condition > 1].index
df_multi = train[train['eeg_id'].isin(idx_condition)].copy()

# 한 행씩 당겨서 eeg_min과 eeg_max 비교하기
df_multi['shift'] = (
    df_multi
    .groupby('eeg_id')['eeg_min']
    .transform('shift', periods= -1, fill_value=np.inf)
)

df_view = df_multi[['target', 'eeg_id', 'eeg_min', 'eeg_max', 'shift', 
                    'spectogram_id', 'spec_min', 'spec_max', 'label_id']]

# 다음 행 min보다 이전 행 max가 큰 경우만 뽑아서 index 기준으로 찾기
idx_shift = df_view.loc[df_view['eeg_max'] > df_view['shift']].index + 1
array_shift = train.loc[idx_shift, 'label_id'].values
df[df['label_id'].isin(array_shift)]
idx_new = df[df['label_id'].isin(array_shift)].index -1

# 조건에 맞게 칼럼에 값 대치하기
df_temp = df.loc[
            idx_new, 
            ['eeg_id', 'eeg_label_offset_seconds', 'spectrogram_label_offset_seconds']
          ].copy()
df_temp.sort_values(by=['eeg_id', 'eeg_label_offset_seconds'], inplace=True)

idx_fix = df_view.loc[df_view['eeg_max'] > df_view['shift'], :].index

train.loc[idx_fix, ['eeg_max', 'spec_max']] = (
    df_temp[['eeg_label_offset_seconds', 'spectrogram_label_offset_seconds']].values
)

train.drop(columns=['label_id'], inplace=True)

train.to_csv('~/kaggle-hms/data/processed/train-processed.csv', index=False)
