import argparse

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='./HEST', help='')
    parser.add_argument('--project_path', type=str, default='./CMRCNet', help='')
    parser.add_argument('--cancer_list', type=list, default=['CMRCNet'], help='')

    parser.add_argument('--save_dir', type=str, default="./checkpoints", help='')
    parser.add_argument('--num_workers', type=int, default=0, help='')

    # 模型相关
    parser.add_argument('--model_name', type=str, default='vit_base_patch32_clip_224', help='')

    parser.add_argument('--image_embedding', type=int, default=3072, help='resnet：2048，vit_base_patch32_clip_224：3072 ')

    parser.add_argument('--spot_embedding', type=int, default=3467, help='')
    parser.add_argument('--pretrained', type=bool, default=False, help='')
    parser.add_argument('--trainable', type=bool, default=True, help='')
    parser.add_argument('--temperature', type=float, default=1.0, help='')
    parser.add_argument('--num_projection_layers', type=int, default=1, help='')
    parser.add_argument('--projection_dim', type=int, default=256, help='resnet：256 ；vit_base_patch32_clip_224:768 ')
    parser.add_argument('--dropout', type=float, default=0.1, help='')

    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--lr', type=float, default=1e-4, help='')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='')
    parser.add_argument('--max_lr', type=float, default=5e-5, help='')

    parser.add_argument('--weight_decay', type=float, default=1e-3, help='')
    parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='')
    parser.add_argument('--patience', type=int, default=2, help='')
    parser.add_argument('--factor', type=float, default=0.5, help='')
    parser.add_argument('--size', type=int, default=224, help='')
    parser.add_argument('--exp_name', type=str, default='result', help='')
    parser.add_argument('--num_epochs', type=int, default=60, help='')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='')
    parser.add_argument('--print_seq', type=int, default=3, help='')


    args = parser.parse_args()
    return args
