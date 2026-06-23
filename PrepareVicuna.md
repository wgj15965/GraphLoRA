## How to Prepare Vicuna Weight

Vicuna is an open-source LLaMA-based LLM that has a performance close to
ChatGPT. We use the **v0 version of Vicuna-7B** in GraphLoRA.

To prepare Vicuna's weight, first download Vicuna's **delta** weight from
[https://huggingface.co/lmsys/vicuna-7b-delta-v0](https://huggingface.co/lmsys/vicuna-7b-delta-v0).
In case you have git-lfs installed (https://git-lfs.com), this can be done by

```
git lfs install
git clone https://huggingface.co/lmsys/vicuna-7b-delta-v0   # 7B, needs at least 12G GPU memory
```

Note that this is not directly the working weight, but the difference between
the working weight and the original weight of LLaMA-7B. (Due to LLaMA's rules,
we cannot distribute the weight of LLaMA.)

Then, you need to obtain the original LLaMA-7B weights in the HuggingFace
format either following the instruction provided by HuggingFace
[here](https://huggingface.co/docs/transformers/main/model_doc/llama) or from
the Internet.

When these two weights are ready, we can use tools from Vicuna's team to create
the real working weight. First, install their library that is compatible with
v0 Vicuna by

```
pip install git+https://github.com/lm-sys/FastChat.git@v0.1.10
```

Then, run the following command to create the final working weight

```
python -m fastchat.model.apply_delta \
    --base   /path/to/llama-7b-hf/ \
    --target /path/to/save/working/vicuna-7b-v1.1/ \
    --delta  /path/to/vicuna-7b-delta-v0/
```

Now you are good to go!
