import XCTest
@testable import MTPLXAppCore

/// A model folder the user chose on this Mac must become a lasting picker
/// row: identity from the path, recognised in every form the path can be
/// written, remembered once, and launched as `--model <path>`.
final class LocalFolderModelTests: XCTestCase {
    private let home = NSHomeDirectory()

    override func setUp() {
        super.setUp()
        unsetenv("MTPLX_APP_DISABLE_LOCAL_MODEL_SCAN")
    }

    // MARK: localFolderModel factory

    func testLocalFolderModelIdentityComesFromTheFolder() throws {
        let option = try XCTUnwrap(
            MTPLXModelOption.localFolderModel(path: "/Volumes/Models/My_Model Folder")
        )
        XCTAssertEqual(option.id, "local-my-model-folder")
        XCTAssertEqual(option.displayName, "My_Model Folder")
        XCTAssertEqual(option.shortName, "My_Model Folder")
        XCTAssertEqual(option.hfModelID, "/Volumes/Models/My_Model Folder")
        XCTAssertEqual(option.localCandidates, ["/Volumes/Models/My_Model Folder"])
        XCTAssertEqual(option.aliases, ["/Volumes/Models/My_Model Folder"])
        XCTAssertTrue(option.isLocalFolder)
        XCTAssertFalse(option.arOnly)
    }

    func testLocalFolderModelMatchesRawTildeAndTrailingSlashForms() throws {
        let expanded = home + "/mtplx-local-test/Foo-Model"
        let option = try XCTUnwrap(MTPLXModelOption.localFolderModel(path: " ~/mtplx-local-test/Foo-Model/ "))

        XCTAssertEqual(option.hfModelID, expanded, "tilde expanded, trailing slash and whitespace dropped")
        XCTAssertEqual(option.localCandidates, [expanded])
        XCTAssertEqual(option.aliases, ["~/mtplx-local-test/Foo-Model/", expanded])
        XCTAssertTrue(option.matches("~/mtplx-local-test/Foo-Model/"))
        XCTAssertTrue(option.matches("~/mtplx-local-test/Foo-Model"))
        XCTAssertTrue(option.matches(expanded))
        XCTAssertTrue(option.matches(expanded + "/"))
        XCTAssertFalse(option.matches("/elsewhere/Other-Model"))

        let fromExpanded = try XCTUnwrap(MTPLXModelOption.localFolderModel(path: expanded))
        XCTAssertEqual(fromExpanded.id, option.id, "one folder, one identity, however it was typed")
        XCTAssertEqual(fromExpanded.hfModelID, option.hfModelID)
    }

    func testLocalFolderModelAcceptsFileURLs() throws {
        let option = try XCTUnwrap(
            MTPLXModelOption.localFolderModel(path: "file:///Volumes/Models/Foo-Model/")
        )
        XCTAssertEqual(option.hfModelID, "/Volumes/Models/Foo-Model")
        XCTAssertEqual(option.id, "local-foo-model")
    }

    func testLocalFolderModelRejectsAnythingThatIsNotAPath() {
        XCTAssertNil(MTPLXModelOption.localFolderModel(path: ""))
        XCTAssertNil(MTPLXModelOption.localFolderModel(path: "   "))
        XCTAssertNil(MTPLXModelOption.localFolderModel(path: "Foo/Bar"))
        XCTAssertNil(MTPLXModelOption.localFolderModel(path: "https://huggingface.co/Foo/Bar"))
        XCTAssertNil(MTPLXModelOption.localFolderModel(path: "optimized-speed"))
        XCTAssertNil(MTPLXModelOption.localFolderModel(path: "/"), "the root is not a model folder")
    }

    func testHuggingFaceParserRefusesFilesystemPaths() {
        // The add field accepts both forms, so the two parsers must never
        // both accept one input: "/Volumes/Foo" used to parse as the repo
        // "Volumes/Foo" once the slash trim ran.
        XCTAssertNil(MTPLXModelOption.customHuggingFaceModel(repoID: "/Volumes/Foo"))
        XCTAssertNil(MTPLXModelOption.customHuggingFaceModel(repoID: "~/Foo/Bar"))
        XCTAssertNil(MTPLXModelOption.normalizedHuggingFaceRepoID("/Volumes/Foo"))
        XCTAssertEqual(MTPLXModelOption.normalizedHuggingFaceRepoID("Foo/Bar"), "Foo/Bar")
        XCTAssertEqual(MTPLXModelOption.normalizedHuggingFaceRepoID("Foo/Bar/"), "Foo/Bar")
        XCTAssertEqual(
            MTPLXModelOption.normalizedHuggingFaceRepoID("https://huggingface.co/Foo/Bar/tree/main"),
            "Foo/Bar"
        )
    }

    // MARK: AppConfiguration.rememberLocalFolderModel

    func testRememberLocalFolderModelRecordsOnePickerRow() {
        var config = MTPLXAppConfiguration()
        config.rememberLocalFolderModel(path: "/Volumes/Models/Foo-Model")
        XCTAssertEqual(config.customModels.map(\.id), ["local-foo-model"])
        XCTAssertEqual(config.customModels[0].displayName, "Foo-Model")
    }

    func testRememberLocalFolderModelDedupesByPath() {
        var config = MTPLXAppConfiguration()
        config.rememberLocalFolderModel(path: "~/mtplx-local-test/Foo-Model")
        config.rememberLocalFolderModel(path: home + "/mtplx-local-test/Foo-Model/")
        config.rememberLocalFolderModel(path: "~/mtplx-local-test/Foo-Model")
        XCTAssertEqual(config.customModels.count, 1, "the same folder in every spelling is one row")
    }

    func testRememberLocalFolderModelDedupesByIdAndTheNewFolderWins() {
        var config = MTPLXAppConfiguration()
        config.rememberLocalFolderModel(path: "/Volumes/Old/Foo-Model")
        config.rememberLocalFolderModel(path: "/Volumes/New/Foo-Model")
        XCTAssertEqual(config.customModels.count, 1)
        XCTAssertEqual(config.customModels[0].hfModelID, "/Volumes/New/Foo-Model")
    }

    func testRememberLocalFolderModelRefusesCatalogInstallDirectories() throws {
        let official = MTPLXModelOption.officialCatalog[0]
        let installDirectory = try XCTUnwrap(official.localCandidates.first)
        var config = MTPLXAppConfiguration()
        config.rememberLocalFolderModel(path: installDirectory)
        config.rememberLocalFolderModel(path: (installDirectory as NSString).expandingTildeInPath)
        XCTAssertTrue(config.customModels.isEmpty, "the catalog row already launches its own install directory")
    }

    func testRememberLocalFolderModelKeepsAnEntryThatAlreadyPointsAtTheFolder() {
        var config = MTPLXAppConfiguration()
        config.rememberForgedModel(brandedName: "Friendly-Foo", localPath: "/tmp/forged/Foo-Model")
        config.rememberLocalFolderModel(path: "/tmp/forged/Foo-Model/")
        XCTAssertEqual(config.customModels.count, 1)
        XCTAssertEqual(config.customModels[0].id, "forged-friendly-foo", "the richer existing row stays")
    }

    func testRememberLocalFolderModelIgnoresNonPaths() {
        var config = MTPLXAppConfiguration()
        config.rememberLocalFolderModel(path: "")
        config.rememberLocalFolderModel(path: "Foo/Bar")
        XCTAssertTrue(config.customModels.isEmpty)
    }

    func testRememberedLocalFolderSurvivesSettingsRoundTrip() throws {
        var config = MTPLXAppConfiguration()
        config.rememberLocalFolderModel(path: "/Volumes/Models/Foo-Model")
        config.model = "/Volumes/Models/Foo-Model"
        let data = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(MTPLXAppConfiguration.self, from: data)
        XCTAssertEqual(decoded.customModels, config.customModels)
        XCTAssertTrue(decoded.customModels[0].matches(decoded.model))
    }

    // MARK: pickerCatalog

    func testPickerCatalogSynthesizesARowForACurrentLocalPath() throws {
        let path = "/Volumes/Models/Foo-Model"
        let catalog = MTPLXModelOption.pickerCatalog(customModels: [], currentModel: path)
        let row = try XCTUnwrap(catalog.first { $0.id == "local-foo-model" })
        XCTAssertTrue(row.matches(path))
        XCTAssertEqual(row.resolvedReference, path, "an absent folder still launches by its path")
        XCTAssertEqual(catalog.filter { $0.matches(path) }.count, 1)
    }

    func testPickerCatalogSynthesizesARowForACurrentTildePath() throws {
        let catalog = MTPLXModelOption.pickerCatalog(
            customModels: [],
            currentModel: "~/mtplx-local-test/Foo-Model"
        )
        let row = try XCTUnwrap(catalog.first { $0.id == "local-foo-model" })
        XCTAssertTrue(row.matches("~/mtplx-local-test/Foo-Model"))
        XCTAssertEqual(row.resolvedReference, home + "/mtplx-local-test/Foo-Model")
    }

    func testPickerCatalogDoesNotSynthesizeALocalRowForHuggingFaceIdsOrCatalogModels() {
        let hf = MTPLXModelOption.pickerCatalog(customModels: [], currentModel: "Foo/Bar")
        XCTAssertTrue(hf.contains { $0.id == "custom-foo--bar" })
        XCTAssertFalse(hf.contains(where: \.isLocalFolder))

        let baseline = MTPLXModelOption.pickerCatalog(customModels: [])
        for current in ["Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed", "qwen38-27b-optimized-speed"] {
            let catalog = MTPLXModelOption.pickerCatalog(customModels: [], currentModel: current)
            XCTAssertFalse(catalog.contains(where: \.isLocalFolder), current)
            XCTAssertFalse(catalog.contains { $0.id.hasPrefix("custom-") }, current)
            XCTAssertEqual(catalog.count, baseline.count, current)
        }
    }

    func testPickerCatalogShowsARememberedFolderOnceAfterSwitchingAway() throws {
        var config = MTPLXAppConfiguration()
        config.rememberLocalFolderModel(path: "/Volumes/Models/Foo-Model")
        config.model = "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"

        let catalog = MTPLXModelOption.pickerCatalog(
            customModels: config.customModels,
            currentModel: config.model
        )
        let rows = catalog.filter { $0.matches("/Volumes/Models/Foo-Model") }
        XCTAssertEqual(rows.count, 1, "the folder stays a row while another model is current")
        XCTAssertEqual(rows.first?.resolvedReference, "/Volumes/Models/Foo-Model")
    }

    func testPickerCatalogFoldsACatalogNamedFolderIntoTheCatalogRow() throws {
        // LM Studio layout: <root>/<org>/<repo>. The basename is the catalog
        // model's folder name, so this is that model at another location.
        let folder = try makeCompleteModelFolder(named: "Qwen3.8-27B-MTPLX-Optimized-Speed")
        defer { try? FileManager.default.removeItem(at: folder.deletingLastPathComponent()) }
        let local = try XCTUnwrap(MTPLXModelOption.localFolderModel(path: folder.path))

        for catalog in [
            MTPLXModelOption.pickerCatalog(customModels: [local]),
            MTPLXModelOption.pickerCatalog(customModels: [], currentModel: folder.path),
        ] {
            XCTAssertFalse(catalog.contains(where: \.isLocalFolder), "no duplicate row for the same model")
            let row = try XCTUnwrap(catalog.first { $0.id == "qwen38-27b-optimized-speed" })
            XCTAssertEqual(row.localCandidates.last, folder.path, "the catalog row learns the folder")
            XCTAssertTrue(row.isInstalled)
            XCTAssertTrue(row.matches(folder.path))
            XCTAssertEqual(catalog.filter { $0.id == row.id }.count, 1)
        }

        var config = MTPLXAppConfiguration()
        config.rememberLocalFolderModel(path: folder.path)
        XCTAssertEqual(config.customModels.map(\.id), ["local-qwen3.8-27b-mtplx-optimized-speed"],
                       "a non-standard location is remembered so the catalog row can find it later")
    }

    // MARK: Display name and launch

    func testDisplayNameForARememberedLocalFolderIsTheFolderName() throws {
        let path = "/Volumes/Models/Foo-Model"
        let option = try XCTUnwrap(MTPLXModelOption.localFolderModel(path: path))
        XCTAssertEqual(MTPLXModelOption.displayName(for: path, customModels: [option]), "Foo-Model")
        XCTAssertEqual(
            MTPLXModelOption.displayName(for: "~/mtplx-local-test/Foo-Model", customModels: [option]),
            "Foo-Model"
        )
    }

    func testDisplayNameForACatalogNamedFolderIsTheCatalogName() throws {
        let path = "/Volumes/Models/Qwen3.8-27B-MTPLX-Optimized-Speed"
        let option = try XCTUnwrap(MTPLXModelOption.localFolderModel(path: path))
        let official = try XCTUnwrap(MTPLXModelOption.officialCatalog.first { $0.id == "qwen38-27b-optimized-speed" })
        XCTAssertEqual(
            MTPLXModelOption.displayName(for: path, customModels: [option]),
            official.displayName,
            "the chrome label agrees with the picker row the folder is folded into"
        )
    }

    func testCommandBuilderLaunchesARememberedFolderAsTheModelPath() throws {
        let folder = try makeCompleteModelFolder(named: "Foo-Model")
        defer { try? FileManager.default.removeItem(at: folder.deletingLastPathComponent()) }
        let fake = folder.appendingPathComponent("mtplx")
        try "#!/bin/sh\n".write(to: fake, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fake.path)

        var config = MTPLXAppConfiguration(executablePath: fake.path, profile: "auto")
        let option = try XCTUnwrap(MTPLXModelOption.localFolderModel(path: folder.path + "/"))
        config.rememberLocalFolderModel(path: folder.path + "/")
        config.model = option.resolvedReference

        XCTAssertEqual(config.model, folder.path)
        let command = try MTPLXCommandBuilder(environment: ["PATH": folder.path])
            .buildServeCommand(configuration: config)
        let modelIndex = try XCTUnwrap(command.arguments.firstIndex(of: "--model"))
        XCTAssertEqual(command.arguments[modelIndex + 1], folder.path)
        XCTAssertFalse(command.arguments.contains("--no-mtp"))
    }

    // MARK: Helpers

    /// A directory that passes `hasCompleteInstall`: config, tokenizer and
    /// a single-file weight set, under a fresh temporary parent.
    private func makeCompleteModelFolder(named name: String) throws -> URL {
        let parent = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-local-folder-\(UUID().uuidString)", isDirectory: true)
        let folder = parent.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try "{}".write(to: folder.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try "{}".write(to: folder.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        try Data([0]).write(to: folder.appendingPathComponent("model.safetensors"))
        return folder
    }
}
