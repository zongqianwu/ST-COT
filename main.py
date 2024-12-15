from collections import Counter
import argparse
import numpy as np
import logging
import torch
import random
import time
import os
from utils import *


def calculate_entropy(prob_dist):
    prob_dist = np.array(prob_dist)
    prob_dist = prob_dist[prob_dist > 0]
    entropy = -np.sum(prob_dist * np.log2(prob_dist))
    return entropy

def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    fix_seed(args.random_seed)
    print("OPENAI_API_KEY:")

    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder(args)

    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()


    sample_n = 3
    total = 0
    correct_list = []
    for i, data in enumerate(dataloader):
        # Answer prediction by generating text ...
        print('*************************')
        print("{}st data".format(i + 1))

        # Prepare question template ...
        x, y = data
        y = y[0].strip()

        pre_list = []  # ori and self-consistency
        pre_two_list = [] # two prediction of cycle
        pre_three_list = []  # three prediction of cycle


        z = "Q: " + x[0] + "\n" + "A: "
        z = z + args.cot_trigger # Let's think step by step


        p_d = decoder.decode(args, z, 1, 1, sample_n)   # response.choices
        p1_tem_list = []  # Q + A + Letsthink + aug infor + the answer isxxx

        for j in range(len(p_d)):
            # Once cycle
            print("=========== {}-th CoT ============".format(j + 1))
            p1 = z + "\n" + p_d[j].message.content + "\n" + args.direct_answer_trigger_for_zeroshot_cot + " "
            p1_tem_list.append(p1)
            pred = decoder.decode(args, p1, 1, 1, 1)
            pred = pred[0].message.content
            print(p1 + pred)  # print Q + A
            pred = answer_cleansing(args, pred)
            pre_list.append(pred)

        # Information entropy
        total_count_pre = len(pre_list)
        frequency_pre = Counter(pre_list)
        probabilities_pre = [frequency_pre[item] / total_count_pre for item in dict.fromkeys(pre_list)]
        entropy_pre = calculate_entropy(probabilities_pre)


        # Judge the current CoT prediction
        if entropy_pre > 0:
            p2_tem_list = []  # Q + A + Letsthink + aug infor + the answer isxxx

            for t in range(len(p1_tem_list)):
                # Two cycle
                tem = p1_tem_list[t] + pre_list[t] + "\n"
                tem = tem  + "\n" + "Based on the above thoughts, reevaluate from alternative perspectives to produce deeper, solution-oriented insights that go beyond prior inferences. Focus on identifying unexplored assumptions or unaddressed challenges in the question context, and propose new reasoning steps that might reveal further implications or innovative solutions."

                p_d2 = decoder.decode(args, tem, 1, 1, 1)

                p2 = z + "\n" + p_d2[0].message.content + "\n" + args.direct_answer_trigger_for_zeroshot_cot + " "
                p2_tem_list.append(p2)
                pred_2 = decoder.decode(args, p2, 1, 1, 1)
                pred_2 = pred_2[0].message.content
                # print(p2 + pred_2)

                pred_2 = answer_cleansing(args, pred_2)
                pre_two_list.append(pred_2)


            # Information entropy
            total_count_pre = len(pre_two_list)
            frequency_pre = Counter(pre_two_list)
            probabilities_pre = [frequency_pre[item] / total_count_pre for item in dict.fromkeys(pre_two_list)]
            entropy_pre = calculate_entropy(probabilities_pre)

            if entropy_pre > 0:
                for t in range(len(p2_tem_list)):
                    # Three cycle
                    tem = p2_tem_list[t] + pre_two_list[t] + "\n"
                    tem = tem  + "\n" + "Based on the above thoughts, reevaluate from alternative perspectives to produce deeper, solution-oriented insights that go beyond prior inferences. Focus on identifying unexplored assumptions or unaddressed challenges in the question context, and propose new reasoning steps that might reveal further implications or innovative solutions."
                    p_d3 = decoder.decode(args, tem, 1, 1, 1)

                    p3 = z + "\n" + p_d3[0].message.content + "\n" + args.direct_answer_trigger_for_zeroshot_cot + " "
                    pred_3 = decoder.decode(args, p3, 1, 1, 1)
                    pred_3 = pred_3[0].message.content
                    # print(p3 + pred_3)

                    pred_3 = answer_cleansing(args, pred_3)
                    pre_three_list.append(pred_3)

                # Information entropy
                total_count_pre = len(pre_three_list)
                frequency_pre = Counter(pre_three_list)
                probabilities_pre = [frequency_pre[item] / total_count_pre for item in dict.fromkeys(pre_three_list)]
                entropy_pre = calculate_entropy(probabilities_pre)

                if entropy_pre > 0:
                    mark = 4  # After 3 iterations of CoT predictions were unstable

                else: mark = 3  # CoT predictions only stabilized in the 3rd iteration
            else: mark = 2  # CoT predictions only stabilized in the 2nd iteration
        else: mark = 1  # CoT prediction stabilized at 1st iteration

        # Determine if the second and third rounds of predictions are empty
        if (len(pre_two_list) == sample_n and all(element == '' for element in pre_two_list)) or (len(pre_three_list) == sample_n and all(element == '' for element in pre_three_list)):
            mark = 4

        # Judging whose predictions to use
        if mark == 1:
            print("======= Once cycle - Final Answer of {}st Data ========".format(i + 1))
            last_pre = max(pre_list, key=pre_list.count)
            print("pre_list: ", pre_list)
            print("pred : {}".format(last_pre))
            last_predition = last_pre
        elif mark == 2:
            print("======= Second cycle - Final Answer of {}st Data ========".format(i + 1))
            last_pre_2 = max(pre_two_list, key=pre_two_list.count)
            print("pre_list: ", pre_list)
            print("pre_two_list: ", pre_two_list)
            print("pred_2 : {}".format(last_pre_2))
            last_predition = last_pre_2
        elif mark == 3:
            print("======= Third cycle - Final Answer of {}st Data ========".format(i + 1))
            last_pre_3 = max(pre_three_list, key=pre_three_list.count)
            print("pre_list: ", pre_list)
            print("pre_two_list: ", pre_two_list)
            print("pre_three_list: ", pre_three_list)
            print("pred_3 : {}".format(last_pre_3))
            last_predition = last_pre_3
        elif mark == 4:
            merged_pre = []
            merged_pre.extend(pre_list)
            merged_pre.extend(pre_two_list)
            merged_pre.extend(pre_three_list)
            merged_pre = [element for element in merged_pre if element != '']
            last_pre_mer = max(merged_pre, key=merged_pre.count)
            print("merged_list: ", merged_pre)
            print("merged_pre : {}".format(last_pre_mer))
            last_predition = last_pre_mer

        print('*************************')
        print("GT : " + y)

        # Checking answer ...
        correct = (np.array([last_predition]) == np.array([y])).sum().item()
        correct_list.append(correct)
        total += 1  # np.array([y]).size(0)

        if (args.limit_dataset_size != 0) and ((i + 1) >= args.limit_dataset_size):
            break
            # raise ValueError("Stop !!")

        # Current Accuracy:
        accuracy_cur = (sum(correct_list) * 1.0 / total) * 100
        print("Current Accuracy: {}%".format(accuracy_cur))

    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    print("Total_Accuracy : {}%".format(accuracy))



def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument(
        "--api_log_file_name", type=str, default=None,
        help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]"
    )

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")

    parser.add_argument(
        "--dataset", type=str, default="aqua",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq",
                 "bigbench_date", "object_tracking", "coin_flip", "last_letters"], help="dataset used for experiment"
    )

    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1],
                        help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")

    parser.add_argument("--max_num_worker", type=int, default=3, help="maximum number of workers for dataloader")

    parser.add_argument(
        "--model", type=str, default="gpt3", choices=["gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl"],
        help="model used for decoding. Note that 'gpt3' are the smallest models."
    )

    parser.add_argument(
        "--method", type=str, default="zero_shot_cot",
        choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot"], help="method"
    )
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1,
        help="A trigger sentence that elicits a model to execute chain of thought"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=128,
        help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32,
        help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=10,
        help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log2/", help="log directory"
    )

    args = parser.parse_args()

    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")

    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger

    args.direct_answer_trigger_for_fewshot = "The answer is"

    if args.cot_trigger_no == 1:
        args.cot_trigger = "Let's think step by step."
    elif args.cot_trigger_no == 2:
        args.cot_trigger = "We should think about this step by step."
    elif args.cot_trigger_no == 3:
        args.cot_trigger = "First,"
    elif args.cot_trigger_no == 4:
        args.cot_trigger = "Before we dive into the answer,"
    elif args.cot_trigger_no == 5:
        args.cot_trigger = "Proof followed by the answer."
    elif args.cot_trigger_no == 6:
        args.cot_trigger = "Let's think step by step in a realistic way."
    elif args.cot_trigger_no == 7:
        args.cot_trigger = "Let's think step by step using common sense and knowledge."
    elif args.cot_trigger_no == 8:
        args.cot_trigger = "Let's think like a detective step by step."
    elif args.cot_trigger_no == 9:
        args.cot_trigger = "Let's think about this logically."
    elif args.cot_trigger_no == 10:
        args.cot_trigger = "Let's think step by step. First,"
    elif args.cot_trigger_no == 11:
        args.cot_trigger = "Let's think"
    elif args.cot_trigger_no == 12:
        args.cot_trigger = "Let's think step by step, how to calculate specific quantities or values based on given scenarios and conditions in each situation." # gsm8k
    elif args.cot_trigger_no == 13:
        args.cot_trigger = "Let's think step by step, how to calculate the quantities of items involved in various scenarios and make the necessary deductions with numbers."  # svamp
    elif args.cot_trigger_no == 14:
        args.cot_trigger = "Let's think step by step, how to calculate the total quantity of objects or items mentioned in the sentences." # addsub
    elif args.cot_trigger_no == 15:
        args.cot_trigger = "Let's think step by step, who flips the coin and how many times it gets reversed before determining if it's still heads up?"
    else:
        raise ValueError("cot_trigger_no is not properly defined ...")

    return args

if __name__ == "__main__":
    main()