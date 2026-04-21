

## 项目结构（Git 跟踪）

```
├── .gitignore
├── README.md
├── configs
│   ├── configs_q
│   │   ├── config_bucket_13_22.yaml
│   │   ├── config_bucket_23plus.yaml
│   │   ├── config_bucket_3.yaml
│   │   ├── config_bucket_4.yaml
│   │   ├── config_bucket_5.yaml
│   │   ├── config_bucket_turn0.yaml
│   │   ├── config_bucket_turn1.yaml
│   │   ├── config_bucket_turn2.yaml
│   │   ├── flagged_words
│   │   │   └── flagged_words_installment.txt
│   │   ├── installment_keywords.txt
│   │   │   └── flagged_words.json
│   │   ├── my_config.yaml
│   │   ├── overal_config.yaml
│   │   └── test_2.yaml
│   └── configs_qa
│       ├── config_bucket_10plus.yaml
│       ├── config_bucket_13_22.yaml
│       ├── config_bucket_3.yaml
│       ├── config_bucket_4.yaml
│       ├── config_bucket_5.yaml
│       ├── config_bucket_turn0.yaml
│       ├── config_bucket_turn1.yaml
│       ├── config_bucket_turn2.yaml
│       ├── flagged_words
│       │   └── flagged_words_installment.txt
│       ├── installment_keywords.txt
│       │   └── flagged_words.json
│       ├── my_config.yaml
│       ├── overal_config.yaml
│       ├── test_2.yaml
│       └── test_qa.yaml
├── env_yaml
│   ├── Generalization_scrubadub_env.yml
│   ├── data-juicer-env.yml
│   └── nlpcda_env.yml
├── resources
│   ├── Homophone.txt
│   ├── Homophone_tab.txt
│   ├── bank.txt
│   └── synonyms.txt
└── scripts
    ├── 00_dataset_process.py
    ├── 01_split_dialogues.py
    ├── 02_split_into_buckets.py
    ├── 03_clean_buckets_with_plots.py
    ├── 04_apply_cleaned_loss_direct.py
    ├── 05_main_augment_add.py
    ├── common
    │   ├── __init__.py
    │   └── augment_utils_add.py
    └── homophone_formatting.py
```
