# Dashboard image: production build + unprivileged Nginx
FROM node:22-alpine AS build

WORKDIR /app

# Copy dependency manifests, including lockfile
COPY package.json package-lock.json ./

# Install exact locked dependencies
RUN npm ci --no-audit --no-fund

# Copy frontend source
COPY . .

# Build production frontend
RUN npm run build

# Production Nginx image
FROM nginxinc/nginx-unprivileged:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s \
  CMD wget -qO- http://127.0.0.1:8080/ >/dev/null || exit 1