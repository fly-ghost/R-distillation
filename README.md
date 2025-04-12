## Setup
```bash
pip install -r requirements.txt
```

## Models
Select a teacher model and a student model in ```models``` dir. You can download models on huggingface.

## Start Distillation
run r_distillation.py
```bash
python r_distillation.py --model="deepseek-small" --tokenizer="deepseek-small" --is_decoder=True --batch_size=8 --lr=1e-4 --dataset="humaneval"
```

## Generate Samples
run main.py
```bash
python main.py --mode=2 --model="deepseek-small" --is_decoder=True --is_peft=True --batch_size=8 --dataset="humaneval"
```

## Try Other Distllation Methods
persd means Personalized Distillation, finetune means Fine-Tuning, standard_distillation means standard distillation.