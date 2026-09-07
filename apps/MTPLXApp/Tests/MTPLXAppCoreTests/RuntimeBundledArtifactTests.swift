import Foundation
import XCTest
@testable import MTPLXAppCore

/// Opt-in release-artifact check using the actual installer, without app UI.
/// The caller supplies an isolated HOME and built wheel/Python resources.
final class RuntimeBundledArtifactTests: XCTestCase {
    func testFreshArtifactInstallAndReuse() throws {
        let inputs = ProcessInfo.processInfo.environment
        guard let wheel = inputs["MTPLX_QA_WHEEL"],
              let python = inputs["MTPLX_QA_PYTHON"],
              let home = inputs["MTPLX_QA_HOME"] else {
            throw XCTSkip("Set MTPLX_QA_WHEEL, MTPLX_QA_PYTHON and an isolated MTPLX_QA_HOME")
        }
        let environment = [
            "HOME": home,
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "MTPLX_BUNDLED_RUNTIME_WHEEL": wheel,
            "MTPLX_APP_BUNDLED_PYTHON": python,
        ]
        let bootstrapper = MTPLXRuntimeBootstrapper(environment: environment)
        let installed = try bootstrapper.installOrUpdate()
        let runtime = installed.deletingLastPathComponent().deletingLastPathComponent()
        XCTAssertTrue(installed.path.hasPrefix(home + "/"))
        let selected = try bootstrapper.selectedBundledRuntimeWheel(
            fallback: URL(fileURLWithPath: wheel),
            python: runtime.appendingPathComponent("bin/python")
        )
        if let expected = inputs["MTPLX_QA_EXPECT_WHEEL"] {
            XCTAssertEqual(selected.lastPathComponent, expected)
        }
        XCTAssertEqual(
            MTPLXRuntimeBootstrapper.recordedWheelFingerprint(runtimeDir: runtime),
            try MTPLXRuntimeBootstrapper.wheelFingerprint(of: selected)
        )
        XCTAssertTrue(bootstrapper.runtimeImportHealthAccepted(installedExecutable: installed))
        let before = try FileManager.default.attributesOfItem(atPath: installed.path)[.modificationDate] as? Date
        XCTAssertEqual(try bootstrapper.installOrUpdate(), installed)
        let after = try FileManager.default.attributesOfItem(atPath: installed.path)[.modificationDate] as? Date
        XCTAssertEqual(before, after, "A healthy matching artifact should be reused")
        let receipt: [String: String] = [
            "installed": installed.path,
            "selectedWheel": selected.path,
            "wheelSHA256": try MTPLXRuntimeBootstrapper.wheelFingerprint(of: selected),
        ]
        let data = try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: home).appendingPathComponent("artifact-install-receipt.json"))
    }
}
