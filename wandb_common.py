import wandb

def login_wandb(api_key_file):
    # API 키 파일에서 키를 읽어오기
    with open(api_key_file, 'r') as file:
        api_key = file.read().strip()
    
    # WandB 로그인
    wandb.login(key=api_key)