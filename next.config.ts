import type { NextConfig } from "next";
import path from 'path';

const nextConfig: NextConfig = {
  /* config options here */

  webpack: (config) => {
    config.resolve.alias = {
      ...(config.resolve.alias || {}),
      '@contents': path.resolve(__dirname, 'src/contents'),
      '@data': path.resolve(__dirname, 'src/data'),
    };
    return config;
  },
};

// next.config.js
module.exports = {
  eslint: {
    ignoreDuringBuilds: true,
  },
};

export default nextConfig;
