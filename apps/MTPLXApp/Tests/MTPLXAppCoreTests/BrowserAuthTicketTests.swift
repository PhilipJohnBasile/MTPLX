import XCTest

@testable import MTPLXAppCore

/// Opening the browser dashboard (or the old web chat) with an API key
/// configured must not put the key in the URL: the app asks the daemon for
/// a one-time ticket and opens the URL it returns. When the ticket cannot
/// be minted for any reason, the previous key-in-query URL is used so the
/// button never goes dead; with no key configured the plain URL opens.
@MainActor
final class BrowserAuthTicketTests: XCTestCase {
    private final class StubURLProtocol: URLProtocol {
        nonisolated(unsafe) static var handler: ((URLRequest) throws -> (HTTPURLResponse, Data))?
        nonisolated(unsafe) static var requests: [URLRequest] = []
        /// When true, `startLoading` never answers: the request hangs until
        /// the session's own timeout fires.
        nonisolated(unsafe) static var hangs = false

        override class func canInit(with request: URLRequest) -> Bool { true }

        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

        override func startLoading() {
            var recorded = request
            // URLSession hands body streams to protocols; capture the bytes
            // so the test can assert on the JSON the app sent.
            if recorded.httpBody == nil, let stream = request.httpBodyStream {
                stream.open()
                var data = Data()
                let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: 4096)
                defer { buffer.deallocate() }
                while stream.hasBytesAvailable {
                    let read = stream.read(buffer, maxLength: 4096)
                    guard read > 0 else { break }
                    data.append(buffer, count: read)
                }
                stream.close()
                recorded.httpBody = data
            }
            Self.requests.append(recorded)
            if Self.hangs { return }
            guard let handler = Self.handler else {
                client?.urlProtocol(self, didFailWithError: URLError(.cannotConnectToHost))
                return
            }
            do {
                let (response, data) = try handler(request)
                client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
                client?.urlProtocol(self, didLoad: data)
                client?.urlProtocolDidFinishLoading(self)
            } catch {
                client?.urlProtocol(self, didFailWithError: error)
            }
        }

        override func stopLoading() {}
    }

    private let ticketURL = URL(string: "http://127.0.0.1:18083/mtplx/browser-auth?ticket=t-123&next=%2Fdashboard%2F")!

    override func setUp() {
        super.setUp()
        StubURLProtocol.handler = nil
        StubURLProtocol.requests = []
        StubURLProtocol.hangs = false
    }

    override func tearDown() {
        StubURLProtocol.handler = nil
        StubURLProtocol.requests = []
        StubURLProtocol.hangs = false
        super.tearDown()
    }

    private func stubSession(timeout: TimeInterval = 3) -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        configuration.timeoutIntervalForRequest = timeout
        configuration.timeoutIntervalForResource = timeout
        return URLSession(configuration: configuration)
    }

    private func makeBackend(apiKey: String?, timeout: TimeInterval = 3) -> MTPLXBackendStore {
        MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(host: "127.0.0.1", port: 18083, apiKey: apiKey),
            browserAuthSession: stubSession(timeout: timeout)
        )
    }

    private func response(_ status: Int, for request: URLRequest) -> HTTPURLResponse {
        HTTPURLResponse(
            url: request.url!,
            statusCode: status,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"]
        )!
    }

    private func queryItems(of url: URL) -> [String: String] {
        let components = URLComponents(url: url, resolvingAgainstBaseURL: false)
        return Dictionary(
            uniqueKeysWithValues: (components?.queryItems ?? []).compactMap { item in
                item.value.map { (item.name, $0) }
            }
        )
    }

    func testTicketURLIsUsedWhenTheDaemonMintsOne() async throws {
        StubURLProtocol.handler = { [ticketURL] request in
            let body = #"{"url": "\#(ticketURL.absoluteString)", "expires_in": 60}"#
            return (self.response(200, for: request), Data(body.utf8))
        }
        let backend = makeBackend(apiKey: "dashboard-secret")

        let url = await backend.browserURL(nextPath: "/dashboard/", plain: backend.baseURL.appendingPathComponent("dashboard"))

        XCTAssertEqual(url, ticketURL)
        XCTAssertNil(queryItems(of: url)["mtplx_api_key"], "the key must not be in the opened URL")
        let request = try XCTUnwrap(StubURLProtocol.requests.first)
        XCTAssertEqual(StubURLProtocol.requests.count, 1)
        XCTAssertEqual(request.httpMethod, "POST")
        XCTAssertEqual(request.url?.path, "/mtplx/browser-auth/ticket")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer dashboard-secret")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Content-Type"), "application/json")
        let body = try JSONSerialization.jsonObject(with: XCTUnwrap(request.httpBody)) as? [String: String]
        XCTAssertEqual(body, ["next": "/dashboard/"])
    }

    func testOlderDaemonAnswering404FallsBackToTheKeyURL() async throws {
        StubURLProtocol.handler = { request in
            (self.response(404, for: request), Data(#"{"detail": "Not Found"}"#.utf8))
        }
        let backend = makeBackend(apiKey: "dashboard-secret")

        let url = await backend.browserURL(nextPath: "/dashboard/", plain: backend.baseURL.appendingPathComponent("dashboard"))

        XCTAssertEqual(url.path, "/mtplx/browser-auth")
        XCTAssertEqual(queryItems(of: url), ["mtplx_api_key": "dashboard-secret", "next": "/dashboard/"])
        XCTAssertEqual(url, backend.browserDashboardURL, "the fallback is exactly the previous URL")
    }

    func testUnreachableDaemonFallsBackToTheKeyURL() async {
        StubURLProtocol.handler = nil // the stub fails with cannotConnectToHost
        let backend = makeBackend(apiKey: "dashboard-secret")

        let url = await backend.browserURL(nextPath: "/", plain: backend.baseURL)

        XCTAssertEqual(url, backend.webChatURL)
        XCTAssertEqual(queryItems(of: url)["mtplx_api_key"], "dashboard-secret")
    }

    func testTicketRequestThatNeverAnswersTimesOutAndFallsBack() async {
        StubURLProtocol.hangs = true
        let backend = makeBackend(apiKey: "dashboard-secret", timeout: 0.5)

        let started = ContinuousClock.now
        let url = await backend.browserURL(nextPath: "/dashboard/", plain: backend.baseURL.appendingPathComponent("dashboard"))
        let elapsed = ContinuousClock.now - started

        XCTAssertEqual(url, backend.browserDashboardURL)
        XCTAssertLessThan(elapsed, .seconds(5), "a silent daemon must not hold the button: \(elapsed)")
        XCTAssertEqual(StubURLProtocol.requests.count, 1)
    }

    func testNoConfiguredKeyOpensThePlainURLWithoutAskingTheDaemon() async {
        StubURLProtocol.handler = { request in
            XCTFail("no ticket request is expected without a key: \(request)")
            return (self.response(200, for: request), Data())
        }
        let backend = makeBackend(apiKey: nil)

        let dashboard = await backend.browserURL(nextPath: "/dashboard/", plain: backend.baseURL.appendingPathComponent("dashboard"))
        let chat = await backend.browserURL(nextPath: "/", plain: backend.baseURL)

        XCTAssertEqual(dashboard.absoluteString, "http://127.0.0.1:18083/dashboard")
        XCTAssertEqual(chat.absoluteString, "http://127.0.0.1:18083")
        XCTAssertTrue(StubURLProtocol.requests.isEmpty)
    }

    func testProductionTicketClientIsBoundedToThreeSeconds() {
        let client = MTPLXAPIClient.browserAuthClient(
            baseURL: URL(string: "http://127.0.0.1:18083")!,
            apiKey: "k"
        )
        XCTAssertEqual(client.session.configuration.timeoutIntervalForRequest, 3)
        XCTAssertEqual(client.session.configuration.timeoutIntervalForResource, 3)
    }
}
