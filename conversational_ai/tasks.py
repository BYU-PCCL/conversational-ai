"""Tasks for training a T5 model.

https://github.com/google-research/text-to-text-transfer-transformer
"""
import functools

import chitchat_dataset as ccc
import t5

from conversational_ai.dataset import chitchat, generic

t5.data.TaskRegistry.add(
    "chitchat_v001_nsp",
    t5.data.Task,
    dataset_fn=functools.partial(
        chitchat.dataset,
        generator=lambda: ({"text": "\n".join(c)} for c in ccc.ConversationDataset()),
        keys=["text"],
        num_train=6952,
    ),
    splits=["train", "validation"],
    text_preprocessor=t5.data.preprocessors.next_sentence_prediction,
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[t5.evaluation.metrics.accuracy],
    sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
)


t5.data.TaskRegistry.add(
    "chitchat_v002_compounding",
    t5.data.Task,
    dataset_fn=functools.partial(
        chitchat.dataset,
        generator=functools.partial(
            chitchat.generate_compounding_conversations,
            # `<` is not in the vocab...
            first_speaker_token="speaker1> ",
            second_speaker_token="speaker2> ",
            end_of_utterance_token=" ",  # TODO: change `end_of_utterance_token`
            prefix="converse: ",
        ),
        keys=["inputs", "targets"],
        num_train=124_990,
    ),
    splits=["train", "validation"],
    text_preprocessor=None,
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[
        t5.evaluation.metrics.accuracy,
        t5.evaluation.metrics.bleu,
        t5.evaluation.metrics.rouge,
    ],
    sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
)

for dataset_name in ["dailydialog", "convai2"]:
    t5.data.TaskRegistry.add(
        f"{dataset_name}_v002_compounding",
        t5.data.Task,
        dataset_fn=functools.partial(
            generic.dataset,
            generator=functools.partial(
                generic.generate_compounding_conversations,
                # `<` is not in the vocab...
                first_speaker_token="speaker1> ",
                second_speaker_token="speaker2> ",
                end_of_utterance_token=" ",  # TODO: change `end_of_utterance_token`
                prefix="converse: ",
            ),
            keys=["inputs", "targets"],
            data_dir=f"./data/{dataset_name}",
        ),
        splits=["train", "validation"],
        text_preprocessor=None,
        postprocess_fn=t5.data.postprocessors.lower_text,
        metric_fns=[
            t5.evaluation.metrics.accuracy,
            t5.evaluation.metrics.bleu,
            t5.evaluation.metrics.rouge,
        ],
        sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
    )

t5.data.TaskRegistry.add(
    "chitchat_v003_prefix_lm",
    t5.data.Task,
    dataset_fn=functools.partial(
        chitchat.dataset,
        generator=functools.partial(
            chitchat.generate_conversations_as_str,
            prefix="",
            suffix="",
            turn_prefixes=["speaker1> ", "speaker2> "],  # `<` is not in the vocab...
            turn_suffix="\t",
        ),
        keys=["text"],
        num_train=6952,
    ),
    splits=["train", "validation"],
    text_preprocessor=t5.data.preprocessors.prefix_lm,
    postprocess_fn=t5.data.postprocessors.lower_text,
    metric_fns=[
        t5.evaluation.metrics.accuracy,
        t5.evaluation.metrics.bleu,
        t5.evaluation.metrics.rouge,
    ],
    sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
)

for dataset_name in ["dailydialog", "convai2"]:
    t5.data.TaskRegistry.add(
        f"{dataset_name}_v003_prefix_lm",
        t5.data.Task,
        dataset_fn=functools.partial(
            generic.dataset,
            generator=functools.partial(
                generic.generate_conversations_as_str,
                prefix="",
                suffix="",
                turn_prefixes=[
                    "speaker1> ",
                    "speaker2> ",
                ],  # `<` is not in the vocab...
                turn_suffix="\t",
            ),
            keys=["text"],
            data_dir=f"./data/{dataset_name}",
        ),
        splits=["train", "validation"],
        text_preprocessor=t5.data.preprocessors.prefix_lm,
        postprocess_fn=t5.data.postprocessors.lower_text,
        metric_fns=[
            t5.evaluation.metrics.accuracy,
            t5.evaluation.metrics.bleu,
            t5.evaluation.metrics.rouge,
        ],
        sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
    )
