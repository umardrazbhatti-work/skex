import json

import pytest

from skex.decode.interface import FROZEN_ENGINE, build_messages, empty_card, generate


def test_smoke_card_is_json():
    raw = generate("doc", arm="smoke", engine="none")
    card = json.loads(raw)
    assert card["task"] == []
    assert card["evidence_spans"] == []
    assert raw == empty_card()


def test_prompt_json_refuses_without_a_model():
    with pytest.raises(RuntimeError, match="loaded model"):
        generate("The paper uses BERT.", arm="prompt_json", engine=FROZEN_ENGINE, instruction="Extract.")


def test_other_engines_are_refused():
    with pytest.raises(ValueError, match="outlines"):
        generate("doc", arm="constrained", engine="xgrammar", model=object(), tokenizer=object())


def test_chat_tensors_reads_a_token_dict_not_shape():
    from skex.decode.interface import chat_tensors

    class TokenDict(dict):
        def __getattr__(self, name):
            raise AttributeError(name)

    ids, mask = chat_tensors(TokenDict(input_ids="IDS", attention_mask="MASK"))
    assert ids == "IDS"
    assert mask == "MASK"
    bare, no_mask = chat_tensors("TENSOR")
    assert bare == "TENSOR"
    assert no_mask is None


def test_bounded_prompt_keeps_a_short_string_and_clips_a_long_one():
    from skex.decode.interface import bounded_prompt

    class FakeTok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(ch) for ch in text]}

        def decode(self, ids, skip_special_tokens=False):
            return "".join(chr(i) for i in ids)

    tok = FakeTok()
    assert bounded_prompt("abcd", tok, 10) == "abcd"
    assert bounded_prompt("abcdef", tok, 3) == "abc"


def test_prompt_keeps_document_separate_from_schema():
    messages = build_messages("Extract a card.", "We use BERT on CoNLL.", '{"type":"object"}')
    assert messages[1]["content"] == "We use BERT on CoNLL."
    assert "BERT" not in messages[0]["content"]
    assert '{"type":"object"}' in messages[0]["content"]
