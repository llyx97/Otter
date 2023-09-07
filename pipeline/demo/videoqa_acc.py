# Compute the acc of videoqa, with human result as ground-truth
import argparse, json, os
import numpy as np
from correlation import score_map, extract_auto_score, flatten

def load_auto_vqa_result(args):
    # Load evaluation results based on Video QA
    root_path = args.auto_result_path
    results, fail_num = {}, {}  # count the number of llm_output with incorrect format
    for model in args.model_names:
        results[model], fail_num[model] = {}, 0
        path = os.path.join(root_path, model)
        for seed in range(args.num_auto_runs):
            fn = f'seed{seed}.json'
            if not os.path.isfile(os.path.join(path, fn)):
                continue
            results[model][seed] = {}
            with open(os.path.join(path, fn), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    result = list(json.loads(line).values())[0]
                    if result['answer'] is None:
                        continue
                    score = extract_auto_score(result['answer'], qtype='binary')
                    if score is None:
                        fail_num[model]+=1
                        score = 0
                    cid = result['caption_id']
                    if cid not in results[model][seed]:
                        results[model][seed][cid] = [score]
                    else:
                        results[model][seed][cid] += [score]
        results[model]['avg'] = {id: np.mean([results[model][seed][id] for seed in results[model] if results[model][seed][id] is not None], axis=0) for id in results[model][0]}
        results[model]['avg'] = {id: score_map(args, scores, score_type='continual').tolist() for id, scores in results[model]['avg'].items()}
    return results

def human_score_map(text):
    if text=='yes':
        return 1
    elif text=='no' or text=='Unclear':
        return 0
    else:
        return -1

def load_human_vqa_result(args):
    results = {model: {} for model in args.model_names}
    with open(os.path.join(args.human_result_path, 'human_vqa_answer_100_199.json'), 'r') as f:
        lines = f.readlines()
        for l in lines:
            result = json.loads(l)
            cid = result['caption_id']
            for model in args.model_names:
                score = human_score_map(result['answer'][model])
                if cid not in results[model]:
                    results[model][cid] = [score]
                else:
                    results[model][cid] += [score]
    return results

def compute_acc(args, auto_results, human_results):
    cids = list(human_results[args.model_names[0]].keys())
    human_results = np.array(flatten([list(human_results[model].values()) for model in human_results]))
    auto_results = {model: {cid: auto_results[model]['avg'][cid] for cid in cids} for model in auto_results}
    auto_results = np.array(flatten([list(auto_results[model].values()) for model in auto_results]))

    # majority_preds = np.ones(len(human_results)) * np.argmax(np.bincount(human_results))   # majority baseline
    majority_pred = np.argmax(np.bincount(human_results + abs(np.min(human_results)))) - abs(np.min(human_results)) # majority baseline
    num_correct, num_total = sum(human_results==auto_results), sum(human_results!=-1)
    acc_majority = sum(human_results==majority_pred)/num_total
    print(f"Acc={100*num_correct/num_total:.3f}, Acc Majority={100*acc_majority:.3f}")

if "__main__" == __name__:
    parser = argparse.ArgumentParser()      
    parser.add_argument('--model_names', nargs='+', default=['cogvideo', 'text2video-zero', 'videofusion', 'ground-truth'])
    parser.add_argument('--qtype', default='binary', choices=['binary', 'three_scale', 'five_scale', 'five_scale_complex'])
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_sample_to_eval', default=-1, type=int)
    parser.add_argument('--num_auto_runs', default=1, type=int, help='number of auto llm evaluation runs')
    parser.add_argument('--auto_result_path', default='answers/score', help='the path of video qa result')
    parser.add_argument('--human_result_path', default='answers/human', help='the path of video qa result')
    args = parser.parse_args()

    auto_results = load_auto_vqa_result(args)
    human_results = load_human_vqa_result(args)
    compute_acc(args, auto_results, human_results)