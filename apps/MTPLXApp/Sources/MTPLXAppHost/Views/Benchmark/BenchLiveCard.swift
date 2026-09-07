import SwiftUI
import Foundation
import MTPLXAppCore

// MARK: - BenchLiveCard
//
// The mid-run card: problem statement, status pill, reasoning stream,
// answer hero, and the 4-cell telemetry strip. Sits on a PanelChrome
// surface (was a duplicated inline chrome ZStack — now routed
// through the shared primitive so all benchmark panels read with the
// same surface tone and elevation).
//
// Jet Chrome pass: status pill (Solving / Paused / Done) reads chrome
// for the running state instead of cool-blue; answer hero swaps to
// the `chromeText` polished-steel treatment when neutral (pre-grade)
// so the hero number reads like the wordmark and the gauge TPS;
// telemetry strip uses a Grid so the 4 cells stay column-aligned
// across widths.

struct BenchLiveCard: View {
    let currentIdx: Int
    let total: Int
    let currentProblem: BenchProblem?
    let extractedAnswer: Int?
    let answerIsGraded: Bool
    let state: BenchRunState
    let currentRequestID: String?
    let verificationState: BenchAnswerVerificationState?
    /// Hard pixel width the card must render inside. Threaded through
    /// to MathProblemRender and ReasoningStreamView so the FlowLayout
    /// + Text wrapping receive an explicit, finite proposal — without
    /// this the inner layouts inherit `.frame(maxWidth: .infinity)`
    /// from the chain and expand to their intrinsic widest layout,
    /// which clips the problem text on both sides of the panel.
    let availableWidth: CGFloat
    let liveTelemetry: AIMELiveTelemetryStore

    // NOTE: this card deliberately does NOT observe `MTPLXBackendStore`.
    // It used to hold `@EnvironmentObject backend` to read live decode /
    // prefill TPS, which meant the WHOLE card — including the expensive
    // LaTeX problem + reasoning renders — re-rendered on every metrics
    // tick (tens of times a second during decode), not just when the text
    // changed. The live telemetry now lives in `BenchLiveTelemetry`, the
    // only piece that needs the backend, so it re-renders alone while the
    // math-heavy body only redraws when its own inputs change.
    @ObservedObject var reasoningDocument: StreamingDocumentStore
    @ObservedObject var answerDocument: StreamingDocumentStore
    @State private var problemExpanded: Bool = false

    private var displayTotal: Int {
        max(total, currentIdx)
    }

    /// Width of the card's *inner* content, after subtracting the
    /// outer padding. Every inner panel pins to this so SwiftUI's
    /// layout system can't propose anything wider.
    private var innerContentWidth: CGFloat {
        max(0, availableWidth - 28)
    }

    private var answerHeroWidth: CGFloat { 168 }
    private var reasoningPanelWidth: CGFloat {
        max(0, innerContentWidth - answerHeroWidth - 14)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            headerRow
            problemBlock
            Divider().overlay(Brand.separator)
            // The reasoning + answer row FILLS the card's remaining height
            // rather than claiming a fixed slice. Combined with the panel's
            // fill-height layout this keeps the whole benchmark on one
            // screen (no outer scroll) while the reasoning trace scrolls
            // inside its own box.
            HStack(alignment: .top, spacing: 14) {
                reasoningPanel
                    .frame(width: reasoningPanelWidth, alignment: .leading)
                answerHero
                    .frame(width: answerHeroWidth, alignment: .leading)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            Divider().overlay(Brand.separator)
            telemetryStrip
        }
        .padding(14)
        .frame(width: availableWidth, alignment: .top)
        .frame(maxHeight: .infinity, alignment: .top)
        .background {
            PanelChrome(cornerRadius: Brand.Radii.l, elevation: Brand.Elevation.mid)
        }
    }

    // MARK: - Header

    @ViewBuilder
    private var headerRow: some View {
        HStack(spacing: 10) {
            if let problem = currentProblem, !problem.problem.isEmpty {
                Text(tr("Problem %lld of %lld", currentIdx, displayTotal))
                    .font(.system(size: 13, weight: .heavy, design: .rounded))
                    .foregroundStyle(Brand.typeBody)
                Text("· \(problem.set) #\(problem.index)")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(Brand.typeSecondary)
            } else {
                Text(tr("Waiting for problem…"))
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(Brand.typeSecondary)
            }
            Spacer()
            statusPill
        }
    }

    @ViewBuilder
    private var statusPill: some View {
        HStack(spacing: 6) {
            switch state {
            case .running:
                if let verificationState {
                    if verificationState.isRunning {
                        Image(systemName: "checkmark.seal")
                            .foregroundStyle(Brand.accentChrome)
                            .font(.system(size: 11, weight: .heavy))
                        Text(tr("Checking"))
                            .font(.system(size: 11, weight: .heavy, design: .monospaced))
                            .tracking(1.1)
                            .foregroundStyle(Brand.accentChrome)
                    } else {
                        Image(systemName: verificationState.verificationIcon)
                            .foregroundStyle(verificationState.verificationTint)
                            .font(.system(size: 11, weight: .heavy))
                        Text(verificationState.verificationLabel)
                            .font(.system(size: 11, weight: .heavy, design: .monospaced))
                            .tracking(1.1)
                            .foregroundStyle(verificationState.verificationTint)
                    }
                } else {
                    ThinkingIndicatorDots(color: Brand.accentChrome)
                    Text(tr("Solving"))
                        .font(.system(size: 11, weight: .heavy, design: .monospaced))
                        .tracking(1.1)
                        .foregroundStyle(Brand.accentChrome)
                }
            case .paused:
                Image(systemName: "pause.fill")
                    .foregroundStyle(Brand.warning)
                    .font(.system(size: 11, weight: .heavy))
                Text(tr("Paused"))
                    .font(.system(size: 11, weight: .heavy, design: .monospaced))
                    .tracking(1.1)
                    .foregroundStyle(Brand.warning)
            case .done:
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(Brand.success)
                Text(tr("Done"))
                    .font(.system(size: 11, weight: .heavy, design: .monospaced))
                    .tracking(1.1)
                    .foregroundStyle(Brand.success)
            default:
                EmptyView()
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background {
            Capsule().strokeBorder(Brand.separatorStrong, lineWidth: Brand.hairline)
        }
    }

    // MARK: - Problem text (math-aware)

    @ViewBuilder
    private var problemBlock: some View {
        if let problem = currentProblem, !problem.problem.isEmpty {
            let cleaned = problemCleaned(problem.problem)
            VStack(alignment: .leading, spacing: 6) {
                MathProblemRender(text: cleaned, expanded: problemExpanded)
                    .equatable()
                    .frame(width: innerContentWidth, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
                if problem.hasAsymptoteFigure || cleaned.count > 320 {
                    HStack(spacing: 8) {
                        if problem.hasAsymptoteFigure, let url = URL(string: problem.source) {
                            Link(destination: url) {
                                HStack(spacing: 4) {
                                    Image(systemName: "photo")
                                        .font(.system(size: 10, weight: .semibold))
                                    Text(tr("Figure on AoPS"))
                                        .font(.system(size: 11, weight: .semibold, design: .rounded))
                                }
                                .foregroundStyle(Brand.accentChrome)
                            }
                        }
                        if cleaned.count > 320 {
                            Button(action: { problemExpanded.toggle() }) {
                                Text(problemExpanded ? tr("Show less") : tr("Show more"))
                                    .font(.system(size: 11, weight: .semibold, design: .rounded))
                                    .foregroundStyle(Brand.typeSecondary)
                            }
                            .buttonStyle(.plain)
                        }
                        Spacer()
                    }
                }
            }
        }
    }

    /// Strips `[asy]...[/asy]` figure blocks so math/text rendering
    /// never tries to render Asymptote source. The "Figure on AoPS"
    /// link surfaces the original figure separately.
    private func problemCleaned(_ raw: String) -> String {
        guard let range = raw.range(of: #"\[asy\][\s\S]*?\[/asy\]"#, options: .regularExpression) else {
            return raw
        }
        var cleaned = raw
        cleaned.replaceSubrange(range, with: "")
        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Reasoning panel

    private var reasoningPanel: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(tr("REASONING"))
                .font(.system(size: 9, weight: .heavy, design: .monospaced))
                .tracking(1.4)
                .foregroundStyle(Brand.typeTertiary)
            ReasoningStreamView(document: reasoningDocument, active: state == .running)
                .frame(width: reasoningPanelWidth, alignment: .leading)
                .frame(maxHeight: .infinity, alignment: .leading)
                .background {
                    RoundedRectangle(cornerRadius: Brand.Radii.s, style: .continuous)
                        .fill(Brand.bgInner.opacity(0.6))
                        .overlay {
                            RoundedRectangle(cornerRadius: Brand.Radii.s, style: .continuous)
                                .strokeBorder(Brand.separator, lineWidth: Brand.hairline)
                        }
                }
        }
    }

    // MARK: - Answer hero

    private var answerHero: some View {
        // The label sits ABOVE the surface, exactly like the REASONING
        // panel — it is a section header, not part of the answer card.
        VStack(alignment: .leading, spacing: 6) {
            Text(tr("FINAL ANSWER"))
                .font(.system(size: 9, weight: .heavy, design: .monospaced))
                .tracking(1.4)
                .foregroundStyle(Brand.typeTertiary)
            VStack(spacing: 6) {
                Spacer(minLength: 0)
                heroAnswerText
                if let problem = currentProblem,
                   let extracted = extractedAnswer,
                   currentResultIsGraded,
                   extracted != problem.answer {
                    Text(tr("expected %lld", problem.answer))
                        .font(.system(size: 10, weight: .medium, design: .monospaced))
                        .foregroundStyle(Brand.typeSecondary)
                        .frame(maxWidth: .infinity, alignment: .center)
                }
                Spacer(minLength: 0)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background {
                RoundedRectangle(cornerRadius: Brand.Radii.s, style: .continuous)
                    .fill(Brand.bgInner.opacity(0.6))
                    .overlay {
                        RoundedRectangle(cornerRadius: Brand.Radii.s, style: .continuous)
                            .strokeBorder(Brand.separator, lineWidth: Brand.hairline)
                    }
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(answerAccessibilityLabel)
        }
    }

    /// Hero answer text. Reads as polished chrome when neutral
    /// (pre-grade), and flips to the semantic success / danger color
    /// once the orchestrator has graded the question.
    @ViewBuilder
    private var heroAnswerText: some View {
        let label = Text(displayAnswer)
            .font(.system(size: 40, weight: .heavy, design: .rounded))
            .monospacedDigit()
            .lineLimit(1)
            .minimumScaleFactor(0.5)
            .contentTransition(.numericText())
            .animation(.smooth(duration: 0.28), value: extractedAnswer)
        if currentResultIsGraded {
            label
                .foregroundStyle(answerColor)
                .animation(.smooth(duration: 0.28), value: answerColor)
        } else {
            label.chromeText()
        }
    }

    private var answerAccessibilityLabel: String {
        guard let extractedAnswer else {
            return tr("Final answer pending")
        }
        if let problem = currentProblem, currentResultIsGraded {
            return extractedAnswer == problem.answer
                ? tr("Final answer %lld, correct.", extractedAnswer)
                : tr("Final answer %lld, wrong, expected %lld.", extractedAnswer, problem.answer)
        }
        return tr("Final answer %lld", extractedAnswer)
    }

    private var displayAnswer: String {
        if let extractedAnswer { return String(extractedAnswer) }
        return "—"
    }

    private var currentResultIsGraded: Bool {
        answerIsGraded && extractedAnswer != nil
    }

    private var answerColor: Color {
        guard let problem = currentProblem,
              let extracted = extractedAnswer,
              currentResultIsGraded else { return Brand.typeBody }
        return extracted == problem.answer ? Brand.success : Brand.danger
    }

    // MARK: - Telemetry strip

    /// The live telemetry is the ONLY part of the card that needs the
    /// backend's high-frequency metrics, so it's isolated in its own view.
    /// Token counts are computed here (cheaply, on the card's ~6 Hz
    /// re-renders) and passed down as plain integers.
    private var telemetryStrip: some View {
        BenchLiveTelemetry(
            currentRequestID: currentRequestID,
            reasoningTokens: reasoningDocument.wordCount,
            answerTokens: answerDocument.wordCount,
            aimeTelemetry: liveTelemetry
        )
    }
}

private extension BenchAnswerVerificationState {
    var verificationLabel: String {
        if disputedAnswer { return tr("Disputed") }
        if correctedAnswer { return tr("Corrected") }
        if hasVerifierAnswer { return tr("Checked") }
        return tr("Unverified")
    }

    var verificationIcon: String {
        if disputedAnswer { return "exclamationmark.triangle.fill" }
        if correctedAnswer { return "arrow.triangle.2.circlepath" }
        if hasVerifierAnswer { return "checkmark.seal.fill" }
        return "exclamationmark.triangle.fill"
    }

    var verificationTint: Color {
        if disputedAnswer { return Brand.warning }
        if correctedAnswer { return Brand.warning }
        if hasVerifierAnswer { return Brand.success }
        return Brand.warning
    }

    var disputedAnswer: Bool {
        guard let resolution else { return false }
        return resolution.contains("disagreed") || resolution.contains("disputed")
    }
}

// MARK: - BenchLiveTelemetry
//
// The decode / prefill / token-count strip. Split out of BenchLiveCard so
// that observing the backend's fast-ticking metrics only invalidates this
// tiny number row — NOT the LaTeX-heavy problem + reasoning renders, which
// previously re-typeset on every metrics tick because the parent card held
// the `@EnvironmentObject`.

private struct BenchLiveTelemetry: View {
    let currentRequestID: String?
    let reasoningTokens: Int
    let answerTokens: Int
    @ObservedObject var aimeTelemetry: AIMELiveTelemetryStore

    @EnvironmentObject private var backend: MTPLXBackendStore
    @State private var heldPrefillTokS: Double? = nil
    @State private var heldPrefillRequestID: String? = nil

    var body: some View {
        let selection = decodeSelection
        let decode = selection.displayValue()
        let prefill = livePrefillTokS
            ?? (heldPrefillRequestID == currentRequestID ? heldPrefillTokS : nil)
        Grid(horizontalSpacing: 22, verticalSpacing: 0) {
            GridRow {
                BenchTelemetryCell(
                    label: tr("DECODE"),
                    value: decode,
                    fractionDigits: 1,
                    unit: decode == nil ? nil : "tok/s",
                    emphasised: true
                )
                BenchTelemetryCell(
                    label: tr("PREFILL"),
                    value: prefill,
                    fractionDigits: 0,
                    unit: prefill == nil ? nil : "tok/s"
                )
                BenchTelemetryCell(
                    label: tr("REASONING"),
                    integer: reasoningTokens,
                    unit: "tok"
                )
                BenchTelemetryCell(
                    label: tr("ANSWER"),
                    integer: answerTokens,
                    unit: "tok"
                )
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .onChange(of: currentRequestID) { _, requestID in
            heldPrefillRequestID = requestID
            heldPrefillTokS = nil
        }
        .onChange(of: livePrefillTokS) { _, value in
            guard let value else { return }
            heldPrefillRequestID = currentRequestID
            heldPrefillTokS = value
        }
        .onAppear {
            recordDisplayedTPSSelection()
        }
    }

    private var livePrefillTokS: Double? {
        if let request = currentAIMERequest(),
           let prefill = request.prefillState?.sanePrefillTokS(maxAxis: 5000) {
            return prefill
        }
        return sanePrefillTokS(from: decodeSelection.metrics)
    }

    private var decodeSelection: AIMEDecodeSelection {
        let latestRequestID = backend.latest?.values["request_id"]?.stringValue
        let inFlightIDs = backend.inFlight.map(\.requestId)
        guard let currentRequestID else {
            return AIMEDecodeSelection(
                source: "nil",
                value: nil,
                metrics: nil,
                currentRequestID: nil,
                selectedRequestID: nil,
                backendLatestRequestID: latestRequestID,
                inFlightIDs: inFlightIDs
            )
        }
        if let metrics = currentAIMEProgressMetrics(currentRequestID: currentRequestID) {
            return AIMEDecodeSelection(
                source: "aime_worker_progress",
                metrics: metrics,
                currentRequestID: currentRequestID,
                selectedRequestID: metrics.values["request_id"]?.stringValue,
                backendLatestRequestID: latestRequestID,
                inFlightIDs: inFlightIDs
            )
        }
        if let request = currentAIMERequest() {
            return AIMEDecodeSelection(
                source: "inflight_exact",
                metrics: MetricsLatest(values: request.lastProgress.values),
                currentRequestID: currentRequestID,
                selectedRequestID: request.requestId,
                backendLatestRequestID: latestRequestID,
                inFlightIDs: inFlightIDs
            )
        }
        if let latest = backend.latest,
           latestRequestID == currentRequestID {
            return AIMEDecodeSelection(
                source: "latest_exact_completed",
                metrics: latest,
                currentRequestID: currentRequestID,
                selectedRequestID: latestRequestID,
                backendLatestRequestID: latestRequestID,
                inFlightIDs: inFlightIDs
            )
        }
        return AIMEDecodeSelection(
            source: "nil",
            value: nil,
            metrics: nil,
            currentRequestID: currentRequestID,
            selectedRequestID: nil,
            backendLatestRequestID: latestRequestID,
            inFlightIDs: inFlightIDs
        )
    }

    private func currentAIMERequest() -> InFlightRequest? {
        guard let currentRequestID else { return nil }
        return backend.inFlight.first { $0.requestId == currentRequestID }
    }

    private func currentAIMEProgressMetrics(currentRequestID: String) -> MetricsLatest? {
        guard let latest = aimeTelemetry.latest else { return nil }
        guard latest.values["request_id"]?.stringValue == currentRequestID else { return nil }
        return latest
    }

    private var telemetryDiagnosticsKey: String {
        guard AIMEDiagnostics.isEnabled else { return "disabled" }
        let selection = decodeSelection
        let rawValue = selection.value.map { String(format: "%.3f", $0) } ?? "nil"
        let displayedValue = selection.displayValue()
            .map { String(format: "%.3f", $0) } ?? "nil"
        return [
            currentRequestID ?? "nil",
            selection.source,
            rawValue,
            displayedValue,
            String(selection.completionTokens ?? -1),
            selection.backendLatestRequestID ?? "nil",
            selection.inFlightIDs.joined(separator: ",")
        ].joined(separator: "|")
    }

    private func recordDisplayedTPSSelection() {
        guard AIMEDiagnostics.isEnabled else { return }
        let selection = decodeSelection
        let displayedValue = selection.displayValue()
        let selectedRequestMismatch = selection.currentRequestID != nil
            && selection.selectedRequestID != nil
            && selection.selectedRequestID != selection.currentRequestID
        let backendLatestRequestMismatch = selection.currentRequestID != nil
            && selection.backendLatestRequestID != nil
            && selection.backendLatestRequestID != selection.currentRequestID
        guard AIMEDiagnostics.shouldRecordCadenced(
            "display_tps_selection",
            intervalS: 0.5,
            tokenCount: selection.completionTokens ?? reasoningTokens + answerTokens,
            identity: selection.currentRequestID,
            force: selectedRequestMismatch || backendLatestRequestMismatch
        ) else { return }
        AIMEDiagnostics.signpost(.displayedTPSSelection)
        var fields = AIMEDiagnostics.fields(
            ("current_request_id", AIMEDiagnostics.string(selection.currentRequestID)),
            ("selected_request_id", AIMEDiagnostics.string(selection.selectedRequestID)),
            ("backend_latest_request_id", AIMEDiagnostics.string(selection.backendLatestRequestID)),
            ("source", .string(selection.source)),
            ("display_decode_tok_s", AIMEDiagnostics.double(displayedValue)),
            ("candidate_decode_tok_s", AIMEDiagnostics.double(selection.value)),
            ("display_decode_suppressed", .bool(selection.suppressesLiveDecode)),
            ("display_decode_warmup_floor_tokens", .int(AIMETelemetryDisplayPolicy.liveDecodeWarmupCompletionTokens)),
            ("sample_completion_tokens", AIMEDiagnostics.int(selection.completionTokens)),
            ("reasoning_tokens", .int(reasoningTokens)),
            ("answer_tokens", .int(answerTokens)),
            ("in_flight_count", .int(selection.inFlightIDs.count)),
            ("in_flight_ids", selection.inFlightIDs.isEmpty ? nil : .string(selection.inFlightIDs.joined(separator: ",")))
        )
        if let metrics = selection.metrics {
            fields.merge(AIMEDiagnostics.metricFields(from: metrics.values, prefix: "selected_")) { _, new in new }
        }
        fields["request_id_mismatch"] = .bool(selectedRequestMismatch)
        fields["backend_latest_request_id_mismatch"] = .bool(backendLatestRequestMismatch)
        AIMEDiagnostics.record("display_tps_selection", fields: fields)
        if selectedRequestMismatch {
            AIMEDiagnostics.record("display_tps_request_id_mismatch", fields: fields)
        }
    }

    private func sanePrefillTokS(from latest: MetricsLatest?) -> Double? {
        guard let latest else { return nil }
        let ceiling = 5000.0

        func sane(_ key: String) -> Double? {
            guard let value = latest.values[key]?.doubleValue,
                  value.isFinite,
                  value > 0,
                  value <= ceiling else {
                return nil
            }
            return value
        }

        return sane("cumulative_prefill_tok_s")
            ?? sane("prefill_tok_s")
            ?? sane("live_prefill_tok_s")
            ?? sane("chunk_prefill_tok_s")
    }
}

private struct AIMEDecodeSelection: Equatable {
    var source: String
    var value: Double?
    var metrics: MetricsLatest?
    var currentRequestID: String?
    var selectedRequestID: String?
    var backendLatestRequestID: String?
    var inFlightIDs: [String]
    var completionTokens: Int?

    init(
        source: String,
        value: Double? = nil,
        metrics: MetricsLatest?,
        currentRequestID: String?,
        selectedRequestID: String?,
        backendLatestRequestID: String?,
        inFlightIDs: [String]
    ) {
        self.source = source
        self.metrics = metrics
        self.currentRequestID = currentRequestID
        self.selectedRequestID = selectedRequestID
        self.backendLatestRequestID = backendLatestRequestID
        self.inFlightIDs = inFlightIDs
        self.value = value ?? metrics.flatMap { $0.liveDecodeTokS ?? $0.displayDecodeTokS }
        self.completionTokens = metrics.flatMap { $0.completionTokens ?? $0.generatedTokens }
    }

    var suppressesLiveDecode: Bool {
        AIMETelemetryDisplayPolicy.suppressesLiveDecode(
            candidate: value,
            source: source,
            completionTokens: completionTokens
        )
    }

    func displayValue() -> Double? {
        AIMETelemetryDisplayPolicy.displayedDecodeTokS(
            candidate: value,
            source: source,
            completionTokens: completionTokens
        )
    }
}
