#### SageMaker Stable Diffusion XL

The SDXL 1.0 Jumpstart provides SDXL optimized for speed and quality, making it the best way to get started if your focus is on inferencing. An instance can be deployed for inferencing, allowing for API use for the image-to-text and image-to-image (including masked inpainting).

The SDXL 1.0 Open Jumpstart is the open SDXL model, ready to be used with custom inferencing code, fine-tuned with custom data, and implemented in any use case. This version does not contain any optimization and may require an instance with more GPU compute.

##### Introduction

Stable Diffusion is a latent diffusion model that generates AI images from text. Instead of operating in the high-dimensional image space, it first compresses the image into the latent space.

Stable Diffusion belongs to a class of deep learning models called diffusion models. They are generative models, meaning they are designed to generate new data similar to what they have seen in training.

Stable Diffusion is a latent diffusion model. Instead of operating in the high-dimensional image space, it first compresses the image into the latent space. The latent space is 48 times smaller so it reaps the benefit of crunching a lot fewer numbers.

##### Concepts

Forward Diffusion
A forward diffusion process adds noise to a training image, gradually turning it into an uncharacteristic noise image. The forward process will turn any clean image into a noise image. Eventually, you won’t be able to tell what the original image was.

Reverse diffusion
Opposite of forward diffision, recovers the clean image from a noisy equivalent. every diffusion process has two parts: (1) drift and (2) random motion. 

To reverse the diffusion, we need to know how much noise is added to an image. The answer is teaching a neural network model to predict the noise added. It is called the noise predictor in Stable Diffusion. 

Image Dimension
The image space is enormous. Think about it: a 512×512 image with three color channels (red, green, and blue) is a 786,432-dimensional space!

Sampler
Aenoising process is called sampling because Stable Diffusion generates a new sample image in each step. The method used in sampling is called the sampler or sampling method.


##### Stable Diffusion XL Inference Parameters

- text: prompt to guide the image generation. Must be specified and should be string.

- width: width of the hallucinated image. If specified, it must be a positive integer divisible by 8.

- height: height of the hallucinated image. If specified, it must be a positive integer divisible by 8. Image size should be larger than 256x256.

- sampler: Available samplers are EulerEDMSampler, HeunEDMSampler,EulerAncestralSampler, DPMPP2SAncestralSampler, DPMPP2MSampler, LinearMultistepSampler

- cfg_scale: A higher cfg_scale results in image closely related to the prompt, at the expense of image quality. If specified, it must be a float. cfg_scale<=1 is ignored. Number between 5 and 15 be used. The default 7 is usually effective for most uses.

- steps: number of denoising steps during image generation. More steps lead to higher quality image. If specified, it must a positive integer.

- seed: fix the randomized state for reproducibility. If specified, it must be an integer. Using the same seed as a previous image without changing other parameters will guide the image to reproduce the same image.

- use_refiner: Refiner is used by defauly with the SDXL model. You can disbale it by using this parameter

- init_image: Image to be used as the starting point.

- image_strength: Indicates extent to transform the reference image. Must be between 0 and 1.

- refiner_steps: Number of denoising steps during image generation for the refiner. More steps lead to higher quality image. If specified, it must a positive integer.

- refiner_strength: Indicates extent to transform the input image to the refiner.

- negative_prompt: guide image generation against this prompt. If specified, it must be a string. It is specified in the - text_prompts with a negative weight.



##### Links

- https://aws.amazon.com/about-aws/whats-new/2023/07/sdxl-1-0-foundation-model-stability-ai-amazon-sagemaker-jumpstart/?nc1=h_ls

- https://aws.amazon.com/blogs/machine-learning/use-stable-diffusion-xl-with-amazon-sagemaker-jumpstart-in-amazon-sagemaker-studio/

- [SageMaker Foundation Models](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-use.html)

- [Pricing](https://aws.amazon.com/sagemaker/pricing/?nc1=h_ls)

- [StabilityAI](https://stability.ai/sdxl-aws-documentation)

- [Git aws-jumpstart-examples](https://github.com/Stability-AI/aws-jumpstart-examples)

- [SDK](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker-runtime.html)

- https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-delete-resources.html

- https://boto3.amazonaws.com/v1/documentation/api/1.26.93/reference/services/sagemaker.html