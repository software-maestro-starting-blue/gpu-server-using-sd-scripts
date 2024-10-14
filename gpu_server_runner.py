import time
import os
import glob
import json
from aws.aws_sqs import receive_messages, delete_message, send_image_success_message, send_image_failure_message
from aws.aws_s3 import upload_image_to_s3

from sdxl_gen_img import setup_parser, setup_logging, main

def extract_message(message_body):
    message = json.loads(message_body)
    return message['diaryId'], message['characterId'], message['prompt'], message['gridPosition']

def get_image_path(diary_id, grid_position):
    folder_path = f"./output/{diary_id}_{grid_position}"
    png_files = glob.glob(os.path.join(folder_path, '*.png'))
    if not png_files:
        return None
    
    return png_files[0]

def delete_output_images():
    pass

def generate_image_sdxl_with_lora(lora_model, prompt, output_dir):
    parser = setup_parser()
    args = parser.parse_args()
    
    args.ckpt = "./models/sd_xl_base_1.0_0.9vae.safetensors"
    args.outdir = output_dir
    args.xformers = True
    args.bf16 = True
    args.W = 512
    args.H = 512
    args.scale = 7.0
    args.sampler = 'dpmsolver++'
    args.network_module = 'networks.lora'
    args.network_weights = lora_model
    args.network_mul = 1.0
    args.steps = 70
    args.batch_size = 1
    args.images_per_prompt = 1
    args.prompt = prompt

    setup_logging(args)
    main(args)

def run():
    while True:

        messages = receive_messages()

        if not messages:
            continue

        for message_body, message_receipt_handle in messages:
            diary_id, character_id, prompt, grid_position = extract_message(message_body)

            lora_model = f"./lora/model_{character_id}.safetensors"
            output_dir = f"./output/{diary_id}_{grid_position}"

            try:
                generate_image_sdxl_with_lora(lora_model, prompt, output_dir)
                image_path = get_image_path(diary_id, grid_position)

                upload_image_to_s3(image_path, diary_id, grid_position)

                send_image_success_message(diary_id, grid_position)
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
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