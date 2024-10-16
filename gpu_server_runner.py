import time
import os
import glob
import json
import logging
from aws.aws_sqs import receive_messages, delete_message, send_image_success_message, send_image_failure_message
from aws.aws_s3 import upload_image_to_s3

from sdxl_gen_img import setup_parser, setup_logging, main

from sdxl_gen_img_preloader import preload, preload_lora

logger = logging.getLogger(__name__)

'''
이 코드에서 Stable Diffusion 모델과 LoRA 모델을 preload합니다.
'''
parser = setup_parser()
args = parser.parse_args()

args.ckpt = "./models/sd_xl_base_1.0_0.9vae.safetensors"
args.xformers = True
args.bf16 = True
args.W = 512
args.H = 512
args.scale = 7.0
args.sampler = 'dpmsolver++'
args.steps = 70
args.batch_size = 1
args.images_per_prompt = 1

setup_logging(args)


args.network_module = ['networks.lora']
lora_paths = {
    2: "./lora/model_2.safetensors",
}
lora_muls = {
    2: 1.0,
}


preload_model_by_character_id = {}
preload_loras = {}

for character_id, lora_path in lora_paths.items():
    args.network_weights = [lora_path]
    args.network_mul = [lora_muls[character_id]]

    dtype, highres_fix, text_encoder1, text_encoder2, vae, unet, tokenizer1, tokenizer2, scheduler_num_noises_per_step, noise_manager, scheduler, device = preload(args)
    
    preload_model_by_character_id[character_id] = (dtype, highres_fix, text_encoder1, text_encoder2, vae, unet, tokenizer1, tokenizer2, scheduler_num_noises_per_step, noise_manager, scheduler, device)
    preload_loras[character_id] = preload_lora(args, vae, text_encoder1, text_encoder2, unet, dtype, device)

def generate_image_sdxl_with_lora(character_id, prompt, output_dir):
    parser = setup_parser()
    args = parser.parse_args()

    args.outdir = output_dir
    args.prompt = prompt

    dtype, highres_fix, text_encoder1, text_encoder2, vae, unet, tokenizer1, tokenizer2, scheduler_num_noises_per_step, noise_manager, scheduler, device = preload_model_by_character_id[character_id]
    networks, network_default_muls, network_pre_calc = preload_loras[character_id]

    main(args, dtype, highres_fix, text_encoder1, text_encoder2, vae, unet, tokenizer1, tokenizer2, scheduler_num_noises_per_step, noise_manager, scheduler, device, networks, network_default_muls, network_pre_calc)


'''
이 코드에서는 메시지를 받고 이미지를 생성한 뒤 S3에 업로드합니다.
'''

def extract_message(message_body):
    message = json.loads(message_body)
    return message['diaryId'], message['characterId'], message['prompt'], message['gridPosition']

def get_image_path(folder_path):
    png_files = glob.glob(os.path.join(folder_path, '*.png'))
    if not png_files:
        return None
    
    return png_files[0]

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def delete_output_images():
    pass

def run():
    while True:

        messages = receive_messages()

        if not messages:
            logger.info("No messages to process.")
            continue

        for message_body, message_receipt_handle in messages:
            diary_id, character_id, prompt, grid_position = extract_message(message_body)
            logger.info("Processing {diary_id}/{grid_position}...")}")

            output_dir = f"./output/{diary_id}_{grid_position}"

            try:
                generate_image_sdxl_with_lora(character_id, prompt, output_dir)
                image_path = get_image_path(output_dir)

                upload_image_to_s3(image_path, diary_id, grid_position)

                delete_file(image_path)

                send_image_success_message(diary_id, grid_position)
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                logger.error(f"An error occurred: {str(e)}")
                send_image_failure_message(diary_id, grid_position)
            finally:
                delete_message(message_receipt_handle)

if __name__ == '__main__':
    try:
        delete_output_images()
        run()
    except KeyboardInterrupt as e:
        print("Shutting down...")
    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        delete_output_images()