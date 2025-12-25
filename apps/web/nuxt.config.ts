// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2024-11-01',
  devtools: { enabled: true },

  modules: [
    '@nuxtjs/tailwindcss',
  ],

  css: ['~/assets/css/main.css'],

  runtimeConfig: {
    public: {
      apiUrl: process.env.NUXT_PUBLIC_API_URL || 'http://localhost:8000',
      wsUrl: process.env.NUXT_PUBLIC_WS_URL || 'ws://localhost:8000',
    },
  },

  app: {
    head: {
      title: 'AI Character',
      meta: [
        { name: 'description', content: 'AI Character Voice Interaction System' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      ],
      htmlAttrs: {
        lang: 'ja',
      },
    },
  },

  typescript: {
    strict: true,
  },
})

