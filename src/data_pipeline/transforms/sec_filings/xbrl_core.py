from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from src.data_pipeline.contracts.sec_filings.documents import DocumentRecord
from src.data_pipeline.contracts.sec_filings.xbrl_core import (
    ContextDefinition,
    XbrlContextRecord,
    XbrlInstanceRecord,
    XbrlUnitRecord,
    context_key_for,
    inline_instance_id,
    source_set_sha256,
    unit_key_for,
    xml_instance_id,
)
from src.data_pipeline.transforms.sec_filings.xbrl_resources import (
    UnitDefinition,
    read_contexts,
    read_units,
)
from src.data_pipeline.transforms.sec_filings.xbrl_sources import (
    INLINE_XBRL,
    FactReference,
    ParsedSource,
    XbrlConflict,
    inline_facts,
    namespace_map,
    schema_refs,
    xml_facts,
)

# Filer-submitted XML instances; any other XML instance of a filing with
# Inline XBRL is EDGAR's rendering of it and must match an Inline target.
_STANDALONE_INSTANCE_TYPE = "EX-101.INS"


@dataclass(frozen=True)
class SourceDocument:
    """A candidate XBRL document: its inventory row and parsed content."""

    document: DocumentRecord
    parsed: ParsedSource


@dataclass(frozen=True)
class LogicalInstance:
    """One logical XBRL instance of a filing with every definition it owns.

    `sources` are its documents ordered by document ordinal; for a merged
    Inline target they include the matching XML representations.
    """

    filing_id: str
    target_name: str | None
    is_inline: bool
    sources: tuple[DocumentRecord, ...]
    schema_refs: tuple[str, ...]
    namespace_map: dict[str, str]
    facts: tuple[FactReference, ...]
    contexts: dict[str, ContextDefinition]
    units: dict[str, UnitDefinition]

    @property
    def instance_id(self) -> str:
        """Deterministic key: the Inline target, or the XML document."""

        if self.is_inline:
            return inline_instance_id(self.filing_id, self.target_name)
        return xml_instance_id(self.filing_id, self.sources[0].document_id)


def _ordered(documents: Sequence[DocumentRecord]) -> tuple[DocumentRecord, ...]:
    return tuple(sorted(documents, key=lambda d: (d.document_ordinal, d.document_id)))


def _checked(instance: LogicalInstance) -> LogicalInstance:
    """Every fact resolves its contextRef and unitRef; schema refs exist."""

    owner = f"instance {instance.instance_id}"
    used_contexts = {fact.context_ref for fact in instance.facts}
    used_units = {fact.unit_ref for fact in instance.facts if fact.unit_ref}
    missing = sorted(used_contexts - instance.contexts.keys())
    if missing:
        raise ValueError(f"{owner}: facts use undefined contexts {missing[:5]}")
    missing = sorted(used_units - instance.units.keys())
    if missing:
        raise ValueError(f"{owner}: facts use undefined units {missing[:5]}")
    if not instance.schema_refs:
        raise ValueError(f"{owner} has no link:schemaRef")
    return instance


def _inline_targets(
    filing_id: str, sources: Sequence[SourceDocument]
) -> list[LogicalInstance]:
    """One instance per Inline target of the filing's Inline document set.

    Facts and `ix:references` name their target (none is the default). With
    one target it owns every context and unit; with several, each owns those
    its facts use, and a resource no target uses is a conflict.
    """

    if not sources:
        return []
    facts = [(s.document, f) for s in sources for f in inline_facts(s.parsed.root)]
    references = [
        (s.document, reference)
        for s in sources
        for reference in s.parsed.root.iter(f"{{{INLINE_XBRL}}}references")
    ]
    names = {fact.target_name for _, fact in facts}
    names |= {reference.get("target") for _, reference in references}
    if not names:
        raise ValueError("Inline XBRL documents hold no fact and no ix:references")
    roots = [source.parsed.root for source in sources]
    contexts, units = read_contexts(roots), read_units(roots)
    namespaces = namespace_map(d for s in sources for d in s.parsed.namespaces)

    targets = []
    for name in sorted(names, key=lambda n: (n is not None, n or "")):
        own = [fact for _, fact in facts if fact.target_name == name]
        if len(names) == 1:
            documents = {
                source.document.document_id: source.document for source in sources
            }
            owned_contexts, owned_units = contexts, units
        else:
            documents = {
                doc.document_id: doc for doc, f in facts if f.target_name == name
            }
            documents |= {
                doc.document_id: doc for doc, r in references if r.get("target") == name
            }
            used_contexts = {fact.context_ref for fact in own}
            used_units = {fact.unit_ref for fact in own if fact.unit_ref}
            owned_contexts = {r: d for r, d in contexts.items() if r in used_contexts}
            owned_units = {r: d for r, d in units.items() if r in used_units}
        target_references = (r for _, r in references if r.get("target") == name)
        targets.append(
            LogicalInstance(
                filing_id=filing_id,
                target_name=name,
                is_inline=True,
                sources=_ordered(list(documents.values())),
                schema_refs=tuple(schema_refs(target_references)),
                namespace_map=namespaces,
                facts=tuple(own),
                contexts=owned_contexts,
                units=owned_units,
            )
        )
    if len(names) > 1:
        unassigned = sorted(
            (contexts.keys() - {f.context_ref for _, f in facts})
            | (units.keys() - {f.unit_ref for _, f in facts if f.unit_ref})
        )
        if unassigned:
            raise XbrlConflict(f"unassigned_inline_resource: {unassigned[:5]}")
    return [_checked(target) for target in targets]


def _xml_instance(filing_id: str, source: SourceDocument) -> LogicalInstance:
    if not source.parsed.is_instance:
        raise ValueError(f"{source.document.document_id} root is not xbrli:xbrl")
    root = source.parsed.root
    return _checked(
        LogicalInstance(
            filing_id=filing_id,
            target_name=None,
            is_inline=False,
            sources=(source.document,),
            schema_refs=tuple(schema_refs([root])),
            namespace_map=namespace_map(source.parsed.namespaces),
            facts=tuple(xml_facts(root)),
            contexts=read_contexts([root]),
            units=read_units([root]),
        )
    )


# Text between tags of canonical segment/scenario XML: typed member values.
_ELEMENT_TEXT = re.compile(r">([^<]+)<")


def _collapse_text(xml: str | None) -> str | None:
    if xml is None:
        return None
    return _ELEMENT_TEXT.sub(lambda m: ">" + " ".join(m.group(1).split()) + "<", xml)


def _same_context(inline: ContextDefinition | None, xml: ContextDefinition) -> bool:
    """Equal definitions, up to whitespace in typed member values: EDGAR's XML
    collapses it (`"Agreements "` becomes `"Agreements"`).
    """

    if inline is None or inline == xml:
        return inline == xml

    def collapsed(context: ContextDefinition) -> ContextDefinition:
        return replace(
            context,
            segment_xml=_collapse_text(context.segment_xml),
            scenario_xml=_collapse_text(context.scenario_xml),
        )

    return collapsed(inline) == collapsed(xml)


def _represents(xml: LogicalInstance, target: LogicalInstance) -> bool:
    """Whether an XML instance is another representation of an Inline target.

    Schema refs and fact references (concept, context, unit) must be equal as
    multisets, and every XML context and unit must be defined identically in
    the target (typed member values up to whitespace); the target may also
    hold resources that no fact uses, which EDGAR's generated XML leaves out.
    """

    def references(instance: LogicalInstance) -> Counter[tuple[str, str, str | None]]:
        return Counter((f.concept, f.context_ref, f.unit_ref) for f in instance.facts)

    return (
        xml.schema_refs == target.schema_refs
        and references(xml) == references(target)
        and all(
            _same_context(target.contexts.get(r), d) for r, d in xml.contexts.items()
        )
        and all(target.units.get(ref) == d for ref, d in xml.units.items())
    )


def build_logical_instances(
    filing_id: str,
    inline_sources: Sequence[SourceDocument],
    xml_sources: Sequence[SourceDocument],
) -> list[LogicalInstance]:
    """Discover a filing's logical instances from its Inline and XML sources.

    Inline XBRL is canonical: each Inline target is an instance, and an XML
    instance representing exactly one target is merged into it. An unmatched
    filer-submitted `EX-101.INS`, or any XML instance of a filing without
    Inline XBRL, is an XML-only instance. Raises XbrlConflict for an XML
    instance matching several targets, or EDGAR-generated XML matching none;
    ValueError for invalid sources.
    """

    targets = _inline_targets(filing_id, inline_sources)
    merged: dict[int, list[DocumentRecord]] = defaultdict(list)
    xml_only = []
    for source in sorted(xml_sources, key=lambda s: s.document.document_ordinal):
        candidate = _xml_instance(filing_id, source)
        matches = [
            i for i, target in enumerate(targets) if _represents(candidate, target)
        ]
        document = source.document
        if len(matches) > 1:
            raise XbrlConflict(f"{document.document_id} matches {len(matches)} targets")
        if matches:
            merged[matches[0]].append(document)
        elif targets and document.document_type != _STANDALONE_INSTANCE_TYPE:
            raise XbrlConflict(
                f"{document.document_id} does not represent the Inline XBRL targets"
            )
        else:
            xml_only.append(candidate)
    instances = [
        replace(target, sources=_ordered([*target.sources, *merged[i]]))
        for i, target in enumerate(targets)
    ]
    return instances + xml_only


def instance_record(
    instance: LogicalInstance, processed_at: datetime
) -> XbrlInstanceRecord:
    """The `xbrl_instances` row of a logical instance."""

    hashes = []
    for document in instance.sources:
        if document.content_sha256 is None:
            raise ValueError(f"{document.document_id} has no content")
        hashes.append((document.document_id, document.content_sha256))
    return XbrlInstanceRecord(
        instance_id=instance.instance_id,
        filing_id=instance.filing_id,
        target_name=instance.target_name,
        is_inline=instance.is_inline,
        source_document_ids=[document.document_id for document in instance.sources],
        schema_refs=list(instance.schema_refs),
        namespace_map=instance.namespace_map,
        fact_count=len(instance.facts),
        context_count=len(instance.contexts),
        unit_count=len(instance.units),
        source_set_sha256=source_set_sha256(hashes),
        processed_at=processed_at,
    )


def context_records(
    instance: LogicalInstance, processed_at: datetime
) -> list[XbrlContextRecord]:
    """The `xbrl_contexts` rows of a logical instance, ordered by context ref."""

    return [
        XbrlContextRecord(
            context_key=context_key_for(instance.instance_id, ref),
            instance_id=instance.instance_id,
            context_ref=ref,
            entity_identifier_scheme=definition.entity_identifier_scheme,
            entity_identifier=definition.entity_identifier,
            period_type=definition.period_type,
            start_date=definition.start_date,
            end_date=definition.end_date,
            instant_date=definition.instant_date,
            segment_xml=definition.segment_xml,
            scenario_xml=definition.scenario_xml,
            context_hash=definition.sha256(),
            processed_at=processed_at,
        )
        for ref, definition in sorted(instance.contexts.items())
    ]


def unit_records(
    instance: LogicalInstance, processed_at: datetime
) -> list[XbrlUnitRecord]:
    """The `xbrl_units` rows of a logical instance, ordered by unit ref."""

    return [
        XbrlUnitRecord(
            unit_key=unit_key_for(instance.instance_id, ref),
            instance_id=instance.instance_id,
            unit_ref=ref,
            unit_kind=definition.unit_kind,
            numerator_measures=list(definition.numerator_measures),
            denominator_measures=list(definition.denominator_measures),
            normalized_expression=definition.expression,
            processed_at=processed_at,
        )
        for ref, definition in sorted(instance.units.items())
    ]
