from pathlib import Path
import sys

path = Path(sys.argv[1]) / 'apps/MTPLXApp/Tests/MTPLXAppCoreTests/MTPLXWorkspaceToolServiceTests.swift'
source = path.read_text()
old = '''        let result = try await object(
            MTPLXChatToolFactory().dispatch(
                name: "write_file",
                argumentsJSON: #"{"path":"bypass.txt","content":"blocked"}"#
            )
        )

        XCTAssertEqual(result["error"] as? String, "unknown_tool")'''
new = '''        let dispatched = await MTPLXChatToolFactory().dispatch(
            name: "write_file",
            argumentsJSON: #"{"path":"bypass.txt","content":"blocked"}"#
        )
        let result = try object(dispatched.resultJSON)

        XCTAssertFalse(dispatched.succeeded)
        XCTAssertEqual(dispatched.failure?.kind, .unknownTool)
        XCTAssertEqual(result["error"] as? String, "unknown_tool")'''
assert source.count(old) == 1
path.write_text(source.replace(old, new))
