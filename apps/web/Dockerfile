# NuxtJS Production Dockerfile
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci

# Copy source and build
COPY . .
RUN npm run build

# Production image
FROM node:20-alpine

WORKDIR /app

# Copy built application
COPY --from=builder /app/.output /app/.output
COPY --from=builder /app/package*.json ./

# Install production dependencies only
RUN npm ci --production

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nuxtjs -u 1001 -G nodejs && \
    chown -R nuxtjs:nodejs /app

USER nuxtjs

EXPOSE 3000

ENV NODE_ENV=production
ENV NUXT_HOST=0.0.0.0
ENV NUXT_PORT=3000

CMD ["node", ".output/server/index.mjs"]

