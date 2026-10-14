/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // This isolated prototype shares dependencies and keeps its own small
  // build outputs. Avoid a second persistent webpack cache on this disk.
  webpack(config) {
    config.cache = false;
    return config;
  },
};

module.exports = nextConfig;
