"""
    Compute correlation with human judgements
"""
import json, os, argparse, statistics, random, math
import pandas as pd
import numpy as np
import scipy.stats as ss
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr
import krippendorff
import matplotlib.pyplot as plt

def flatten(nest_list:list):
    return [j for i in nest_list for j in flatten(i)] if isinstance(nest_list, list) else [nest_list]

def compute_correlation(auto_results, manual_results, compute_cohen=True):
    corr_kendall = {}
    for variant in ['c', 'b']:
        tau = ss.kendalltau(ss.rankdata(auto_results), ss.rankdata(manual_results), variant=variant)
        corr_kendall[f"Kendall-{variant}"] = tau.correlation

    if compute_cohen:
        corr_cohen = cohen_kappa_score(auto_results, manual_results)
    else:
        corr_cohen = 0
    
    df = pd.DataFrame({'auto_eval': auto_results, 'manual_eval': manual_results})
    kripp_alpha = krippendorff.alpha(reliability_data=df.values, level_of_measurement='nominal')

    spearman_coef, p_value = spearmanr(auto_results, manual_results)
    return corr_kendall, corr_cohen, kripp_alpha, spearman_coef, p_value

def load_single_eval_results(result_path, limit=10000):
    """
        load the evaluation results of a single t2v models
    """
    eval_results = {}
    with open(result_path, 'r') as f:
        lines = f.readlines()[:limit]
        for line in lines:
            eval_results.update(json.loads(line))
    # sort the eval_results according to the data id
    if 'manual_eval' in result_path:
        eval_results = {int(id): value for id, value in eval_results.items()}
        eval_results = dict(sorted(eval_results.items()))
        eval_results = {str(id): value for id, value in eval_results.items()}
    return eval_results

def find_latest_eval_result(prefix, root_path='.'):
    eval_files = [os.path.join(root_path, f) for f in os.listdir(root_path) if f.startswith(prefix) and f.endswith('.json')]
    if len(eval_files)==0:
        return
    latest_file = max(eval_files, key=os.path.getctime)
    return latest_file

def load_multi_eval_results(model_names=['cogvideo', 'text2video-zero', 'damo-text2video'], root_path='manual_eval_results', prefix='manual_eval_results'):
    """
        load the evaluation results of a multiple t2v models, from a single human evaluator
    """
    eval_results = {}
    for m_name in model_names:
        result_path = find_latest_eval_result(prefix=f"{prefix}_{m_name}", root_path=root_path)
        eval_results[m_name] = load_single_eval_results(result_path)
    return eval_results

def load_multi_human_results(human_paths, model_names):
    """
        Load evaluation results from multiple human evaluators
    """
    manual_results = {}
    for i, human_path in enumerate(human_paths):
        manual_results[f'human{i}'] = load_multi_eval_results(root_path=human_path, model_names=model_names)

    manual_results['avg'] = {model: {} for model in model_names}
    for model in model_names:
        for id in manual_results[f'human0'][model]:
            manual_results['avg'][model][id] = {}
            for key in manual_results[f'human0'][model][id]:
                if key=='video_id':
                    manual_results['avg'][model][id][key] = manual_results[f'human0'][model][id][key]
                elif key=='fine-grained_alignment':
                    manual_results['avg'][model][id][key] = {}
                    for dim in manual_results['human0'][model][id][key]:
                        avg_score = np.mean([manual_results[f'human{i}'][model][id][key][dim] for i in range(len(human_paths))])
                        manual_results['avg'][model][id][key][dim] = avg_score
                else:
                    avg_score = np.mean([manual_results[f'human{i}'][model][id][key] for i in range(len(human_paths))])
                    manual_results['avg'][model][id][key] = avg_score
    return manual_results['avg']

def extract_auto_score(text, qtype='binary'):
    # Map the textual evaluation to numerical scores
    if qtype=='binary':
        if text.split(',')[0].lower()=='yes':
            return 1
        elif text.split(',')[0].lower()=='no':
            return 0
        else:
            print(f"Fail to produce numerical score for this example: \n---------------\n{text}\n---------------------\n")
            # return random.choice([0,1]) # randomly choose a score
            return
        
    if qtype=='three_scale':
        if 'no match' in text.lower() or 'not match' in text.lower():
            return 0
        elif 'partially' in text.lower():
            return 1
        elif 'match' in text.lower():
            return 2
        else:
            print(f"Fail to produce numerical score for this example: \n---------------\n{text}\n---------------------\n")
            # return random.choice([0,1,2])   # randomly choose a score
            return

def load_single_auto_results(args, path):
    results = {}
    for seed in range(args.num_auto_runs):
        results[seed] = {}
        fn = f'seed{seed}.json'
        with open(os.path.join(path, fn), 'r') as f:
            lines = f.readlines()
            for line in lines:
                result = json.loads(line)
                id, llm_output = int(list(result.keys())[0]), list(result.values())[0]
                if llm_output is None:
                    continue
                score = extract_auto_score(llm_output, qtype=args.qtype)
                if score is None:
                    score = 0
                results[seed][id] = score
    # results['ensemble'] = {id: statistics.mode([results[seed][id] for seed in range(args.num_auto_runs)]) for id in results[0]}    # ensemble the evaluation results of multiple seeds by majority voting

    # sample a subset of evaluation results
    if len(results[args.num_auto_runs-1])!=len(results[0]):
        sampled_ids = list(results[5].keys())
        for seed in range(5):
            results[seed] = {id: results[seed][id] for id in sampled_ids}

    results['avg'] = {id: np.mean([results[seed][id] for seed in range(args.num_auto_runs) if results[seed][id] is not None]) for id in results[0]}    # average the evaluation results of multiple seeds, ignoring textual evaluations that fails to map to scores
    return results

def load_multi_auto_results(args, root_path):
    """
        Load auto evaluation results of multiple T2V models
    """ 
    results = {}   
    for model in args.model_names:
        path = os.path.join(root_path, model, f'qtype_{args.qtype}_AnswerFormat_False_ForceSimple_False')
        results[model] =  load_single_auto_results(args, path)
    return results


def rand_eval_corr(args, manual_results):
    """
        Correlation between manual and random evaluation
    """ 
    if args.qtype=='binary':
        rand_score_pool = [0,1]
    elif args.qtype=='three_scale':
        rand_score_pool=[0,1,2]
    elif 'five_scale' in args.qtype:
        rand_score_pool=[0,1,2,3,4]
    rand_results = [[random.choice(rand_score_pool) for _ in range(len(manual_results))] for _ in range(args.num_auto_runs)]
    avg_rand_results = np.mean(rand_results, axis=0)

    print('\nCorrelation between manual and random evaluation')
    corr_kendall, corr_cohen, kripp_alpha, spearman_coef, p_value = compute_correlation(avg_rand_results, manual_results, compute_cohen=False)
    print(f"kendall_correlation={corr_kendall['Kendall-c']:.3f}, cohen_correlation={corr_cohen:.3f}, kripp_alpha={kripp_alpha:.3f}, spearman={spearman_coef:.3f}, p_value={p_value}")
    
def score_map(args, scores):
    new_scores = []
    for s in scores:
        if args.qtype=='binary':
            if s<=4:
                new_scores.append(0)
            else:
                new_scores.append(1)
        elif args.qtype=='three_scale':
            if s<=2:
                new_scores.append(0)
            elif s>2 and s<=4:
                new_scores.append(1)
            else:
                new_scores.append(2)
    return np.array(new_scores)

def compute_acc(args, manual_results, auto_results):
    """
        Compute the accuray of llm evaluation, after mapping manual evaluation to the same scales of llm evaluation
    """ 
    accuracy = {}
    manual_results = score_map(args, manual_results)
    majority_preds = np.ones(len(manual_results)) * np.argmax(np.bincount(manual_results))   # majority baseline
    acc_majority = sum(majority_preds==manual_results)/len(manual_results)
    for seed in range(args.num_auto_runs):
        tmp_auto_results = np.array(flatten([list(auto_results[model][seed].values()) for model in args.model_names]))
        accuracy[seed] = sum(manual_results==tmp_auto_results)/len(manual_results)
    acc_avg = np.mean(list(accuracy.values()))
    acc_std = np.std(list(accuracy.values()))
    return acc_avg, acc_std, acc_majority

def auto_eval_corr(args, auto_results, manual_results, is_plot=False):
    """
        Correlation between manual and auto evaluation
    """
    print('\nCorrelation Manual-Auto Evaluation')
    corr = {'kendall': [], 'spearman': []}
    for key in auto_results[args.model_names[0]]:
        tmp_auto_results = flatten([list(auto_results[model][key].values()) for model in args.model_names])
        corr_kendall, corr_cohen, kripp_alpha, spearman_coef, p_value = compute_correlation(tmp_auto_results, manual_results, compute_cohen=False)
        if str(key).isdigit():
            corr['kendall'].append(corr_kendall['Kendall-c'])
            corr['spearman'].append(spearman_coef)
        print(f'Correlation between manual and {key} LLM evaluation')
        print(f"kendall_correlation={corr_kendall['Kendall-c']:.3f}, cohen_correlation={corr_cohen:.3f}, kripp_alpha={kripp_alpha:.3f}, spearman={spearman_coef:.3f}, p_value={p_value}")
        if is_plot and key=='avg':
            fig, ax = plt.subplots()
            ax.scatter(tmp_auto_results, manual_results, c='blue')
            ax.set_xlabel("Auto Metric")
            ax.set_ylabel("Human")
            plt.savefig(f"correlation.png")
    print('Avg correlation')
    print(f"kendall={np.mean(corr['kendall']):.3f}({np.std(corr['kendall']):.3f}), spearman={np.mean(corr['spearman']):.3f}({np.std(corr['spearman']):.3f})")

def auto_eval_corr_fg(args, auto_results, manual_results, id_under_cont):
    """
        Correlation between fine-grained manual and auto evaluation
    """
    print('\nCorrelation Fine-grained Manual-Auto Evaluation')
    for cont_dim, ids in id_under_cont.items():
        tmp_auto_results = flatten([[result[cont_dim] for id, result in auto_results[model]['avg'].items() if id in ids] for model in args.model_names])
        tmp_manual_results = flatten([[result['fine-grained_alignment'][cont_dim] for id, result in manual_results[model].items() if int(id) in ids] for model in args.model_names])
        corr_kendall, corr_cohen, kripp_alpha, spearman_coef, p_value = compute_correlation(tmp_auto_results, tmp_manual_results, compute_cohen=False)
        print(f"Dimension={cont_dim}, kendall_correlation={corr_kendall['Kendall-c']:.3f}, cohen_correlation={corr_cohen:.3f}, kripp_alpha={kripp_alpha:.3f}, spearman={spearman_coef:.3f}, p_value={p_value}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def load_auto_fg_results(args, root_path):
    """
        Load auto fine-grained evaluation results of multiple T2V models
    """ 
    results = {}
    id_under_cont = {'color': [], 'quantity': [], 'camera view': [], 'speed': [], 'motion direction': [], 'event order': []} # ids under control dimensions
    for model in args.model_names:
        results[model] = {}
        path = os.path.join(root_path, model, 'fine_grained_eval')
        for seed in range(args.num_auto_runs):
            results[model][seed] = {}
            fn = f'seed{seed}.json'
            with open(os.path.join(path, fn), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    result = json.loads(line)
                    id = int(list(result.keys())[0])
                    if result[str(id)] is None:
                        continue
                    results[model][seed][id] = {}
                    for cont_dim, llm_output in result[str(id)].items():
                        score = extract_auto_score(llm_output, qtype=args.qtype)
                        results[model][seed][id][cont_dim] = score
                        if seed==0 and model==args.model_names[0] and cont_dim in id_under_cont:
                            id_under_cont[cont_dim].append(id)

        results[model]['avg'] = {}
        for id in results[model][0]:
            results[model]['avg'][id] = {cont_dim: np.mean([results[model][seed][id][cont_dim] for seed in range(args.num_auto_runs) if results[model][seed][id][cont_dim] is not None]) for cont_dim in results[model][0][id]}
    return results, id_under_cont

def load_vqa_result(args, root_path):
    # Load evaluation results based on Video QA
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
            results[model][seed] = {cid: np.mean(scores) for cid, scores in results[model][seed].items()}
        results[model]['avg'] = {id: np.mean([results[model][seed][id] for seed in results[model] if results[model][seed][id] is not None]) for id in results[model][0]}
    return results

if "__main__" == __name__:
    parser = argparse.ArgumentParser()      
    parser.add_argument('--model_names', nargs='+', default=['cogvideo', 'text2video-zero', 'videofusion', 'ground-truth'])
    parser.add_argument('--qtype', default='three_scale', choices=['binary', 'three_scale', 'five_scale', 'five_scale_complex'])
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_sample_to_eval', default=-1, type=int)
    parser.add_argument('--num_auto_runs', default=5, type=int, help='number of auto llm evaluation runs')
    args = parser.parse_args()

    set_seed(args.seed)
    manual_result_paths = [f'manual_eval_results/human{i}' for i in range(3)]
    manual_results = load_multi_human_results(manual_result_paths, args.model_names)
    # auto_fg_results, id_under_cont = load_auto_fg_results(args, root_path='answers')
    # auto_results = load_multi_auto_results(args, root_path='answers')
    auto_vqa_results = load_vqa_result(args, root_path='answers')
    # The model-level alignment score
    for model in auto_vqa_results:
        auto_score = np.mean(list(auto_vqa_results[model]['avg'].values()))
        manual_score = np.mean([manual_results[model][id]['alignment'] for id in manual_results[model]])
        print(f"T2V model: {model}, Auto alignment={auto_score}, Manual alignment={manual_score}")

    # # Sample a subset of evaluation results
    # if args.num_sample_to_eval>0:
    #     num_all_sample = len(manual_results[args.model_names[0]])
    #     sampled_ids = random.sample(list(range(num_all_sample)), args.num_sample_to_eval)
    #     for model in args.model_names:
    #         manual_results[model] = {k:v for k,v in manual_results[model].items() if int(k) in sampled_ids}
    #         for seed in range(args.num_auto_runs):
    #             auto_results[model][seed] = {k:v for k,v in auto_results[model][seed].items() if int(k) in sampled_ids}
    #         auto_results[model]['avg'] = {k:v for k,v in auto_results[model]['avg'].items() if int(k) in sampled_ids}
    #         print(model, len(auto_results[model]['avg']), len(manual_results[model]))
    # elif len(auto_results[args.model_names[0]])!=len(manual_results[args.model_names[0]]):  # The auto evaluation results have already been sampled during evaluation
    #     for model in args.model_names:
    #         sampled_ids = list(auto_results[model][0].keys())
    #         # print(manual_results[model].keys(), len(manual_results[model]), len(auto_results[model][0]))
    #         manual_results[model] = {str(id): manual_results[model][str(id)] for id in sampled_ids}

    # auto_eval_corr_fg(args, auto_fg_results, manual_results, id_under_cont)  

    # Merge results of different models into lists
    manual_results = flatten([[manual_results[model][id]['alignment'] for id in manual_results[model]] for model in args.model_names])

    # auto_eval_corr(args, auto_results, manual_results)
    auto_eval_corr(args, auto_vqa_results, manual_results, is_plot=False)
    rand_eval_corr(args, manual_results)


    acc_avg, acc_std, acc_majority = compute_acc(args, manual_results, auto_vqa_results)
    print(f"\nAcc={acc_avg:.3f}, Std={acc_std:.3f}, Majority Baseline={acc_majority:.3f}")