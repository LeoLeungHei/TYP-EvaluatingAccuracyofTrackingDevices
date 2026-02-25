const { createProxyMiddleware } = require("http-proxy-middleware");

module.exports = function (app) {
  // SSE streaming endpoint – must bypass compression / response buffering
  app.use(
    "/api/stream",
    createProxyMiddleware({
      target: "http://localhost:5000",
      changeOrigin: true,
      // Let the Flask response flow straight through without buffering
      selfHandleResponse: false,
      onProxyRes(proxyRes) {
        proxyRes.headers["cache-control"] = "no-cache";
        proxyRes.headers["content-type"] = "text/event-stream";
        proxyRes.headers["x-accel-buffering"] = "no";
      },
    })
  );

  // All other API calls
  app.use(
    "/api",
    createProxyMiddleware({
      target: "http://localhost:5000",
      changeOrigin: true,
    })
  );
};
