

## self-boostrapping

```bash
# 处理所有文件
PYTHONPATH=$(pwd) python run_verify_on_BoN_accelerate.py \
    --local_dir "inference_results/infer_result_gen1" \
    --file_start -1 --file_end -1 \
    --max_workers_per_batch 256 \
    --batch_size 2000
```

## 弃用

这个 repo 用于跑 verify 数据。环境可以先用 openrlhf。

运行 BoN 验证程序（应该还有需要修改的地方，跟之前的数据格式耦合比较多）

```bash
PYTHONPATH=$(pwd) python run_verify_on_BoN.py
```
需要注意或修改的参数
* `all_file_list = glob('/aifs4su/mmcode/codeclm/o1/OpenO1_SFT_ultra_BoN_rewarded/*/*.jsonl')` BoN 输出结果保存的 jsonl 文件
* 最后会输出到同一个目录下，但后缀改为 `_with_correctness.jsonl`
* `output_path = '/aifs4su/mmcode/codeclm/o1/OpenO1_SFT_ultra_BoN_rewarded/uid_to_correct.pkl'` 最后会保存一个 uid-正确次数 的 pkl，感觉好像用不着？
* 调用了 `process_entries()` 函数，输入 list of entries （就是 jsonl 读进来的数据），用来获取 verifier 结果，其中
    * `output_key="response"` 这个参数对应保存 ground truth 的 key
    * `model_output_key="decoded_prompt"`，这个参数对应模型回复。


根据 verify 分数合并数据，并对每条 uid sample 数据。注意这一步会通过 `is_legal_cot_split_boxed` 去掉上一步里面 verifier 认为正确，但是格式不对的回复。

```bash
python sample_and_merge_verified_data.py
```