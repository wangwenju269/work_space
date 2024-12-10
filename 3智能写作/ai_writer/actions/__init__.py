from metagpt.ext.writer.actions.filter_related_docs import RelateFilter
from metagpt.ext.writer.actions.gen_keywords import GenKeyword
from metagpt.ext.writer.actions.gen_summary import GenSummary
from metagpt.ext.writer.actions.refine_context import Refine
from metagpt.ext.writer.actions.write_guide import WriteGuide
from metagpt.ext.writer.actions.write_subsection import WriteSubsection
from metagpt.ext.writer.actions.trans_query import TransQuery
ACTIONS = {
    "RelateFilter": RelateFilter,
    "GenKeyword": GenKeyword,
    "GenSummary": GenSummary,
    "Refine": Refine,
    "WriteGuide": WriteGuide,
    "WriteSubsection": WriteSubsection,
    "TransQuery":TransQuery
}
