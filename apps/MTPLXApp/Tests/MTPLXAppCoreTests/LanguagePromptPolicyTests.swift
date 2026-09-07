import XCTest
import Foundation
@testable import MTPLXAppCore

/// The one-time language prompt: who is asked, and that the stamp that
/// silences it survives the settings file round trip.
final class LanguagePromptPolicyTests: XCTestCase {
    private let stamp = Date(timeIntervalSinceReferenceDate: 800_000_000)

    func testFreshInstallIsNotOfferedThePrompt() {
        // Still inside onboarding: the Language step is the prompt.
        let config = MTPLXAppConfiguration()
        XCTAssertNil(config.onboardingCompletedAt)
        XCTAssertFalse(config.shouldOfferLanguagePrompt)
    }

    func testInstallThatOnboardedBeforeTheLanguageStepIsOfferedOnce() {
        var config = MTPLXAppConfiguration(onboardingCompletedAt: stamp)
        XCTAssertTrue(config.shouldOfferLanguagePrompt, "onboarded, never asked")

        config.languagePromptCompletedAt = Date()
        XCTAssertFalse(config.shouldOfferLanguagePrompt, "asked once is asked")
    }

    func testOnboardingStampsBothDatesSoNewInstallsAreNeverAskedTwice() {
        var config = MTPLXAppConfiguration()
        let finished = Date()
        config.onboardingCompletedAt = finished
        config.languagePromptCompletedAt = finished
        XCTAssertFalse(config.shouldOfferLanguagePrompt)
    }

    func testLegacySettingsFileWithoutTheKeyDecodesAsNeverAsked() throws {
        let json = """
        {"onboarding_completed_at": 800000000.0}
        """
        let config = try JSONDecoder().decode(MTPLXAppConfiguration.self, from: Data(json.utf8))
        XCTAssertEqual(config.onboardingCompletedAt, stamp)
        XCTAssertNil(config.languagePromptCompletedAt)
        XCTAssertTrue(config.shouldOfferLanguagePrompt)
    }

    func testStampRoundTripsThroughTheSettingsCodec() throws {
        var config = MTPLXAppConfiguration(onboardingCompletedAt: stamp)
        config.languagePromptCompletedAt = stamp.addingTimeInterval(60)
        let data = try JSONEncoder().encode(config)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertNotNil(object["language_prompt_completed_at"], "the key the settings file carries")

        let decoded = try JSONDecoder().decode(MTPLXAppConfiguration.self, from: data)
        XCTAssertEqual(decoded.languagePromptCompletedAt, stamp.addingTimeInterval(60))
        XCTAssertFalse(decoded.shouldOfferLanguagePrompt)
    }

    func testAWrongTypedStampDegradesToNeverAskedNotToAnUnreadableFile() throws {
        let json = """
        {"onboarding_completed_at": 800000000.0, "language_prompt_completed_at": "yesterday"}
        """
        let config = try JSONDecoder().decode(MTPLXAppConfiguration.self, from: Data(json.utf8))
        XCTAssertNil(config.languagePromptCompletedAt)
        XCTAssertTrue(config.shouldOfferLanguagePrompt, "asking once more beats losing the settings file")
    }
}
