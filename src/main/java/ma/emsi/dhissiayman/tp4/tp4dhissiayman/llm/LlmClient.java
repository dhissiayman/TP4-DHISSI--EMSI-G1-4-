package ma.emsi.dhissiayman.tp4.tp4dhissiayman.llm;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import jakarta.enterprise.context.Dependent;
import ma.emsi.dhissiayman.tp4.tp4dhissiayman.assistant.Assistant;

import java.io.Serializable;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

@Dependent
public class LlmClient implements Serializable {

    private static final long serialVersionUID = 1L;

    // --------- ÉTAT SÉRIALISABLE ---------
    private String systemRole;

    // --------- COMPOSANTS NON SÉRIALISABLES ---------
    private transient ChatMemory chatMemory;
    private transient Assistant assistant;
    private transient ChatModel model;
    private transient String apiKey;

    // --------- COMPOSANTS RAG PARTAGÉS (UNE SEULE INGESTION POUR TOUTE L'APPLI) ---------
    private static transient RetrievalAugmentor retrievalAugmentor;
    private static transient boolean ragInitialized = false;

    // =========================
    //  RAG : ingestion des PDFs
    // =========================

    private static EmbeddingStore<TextSegment> ingestPdfAsEmbeddingStore(
            Class<?> resourceOwner,
            String resourceName,
            EmbeddingModel embeddingModel) {

        try {
            URL resource = resourceOwner.getClassLoader().getResource(resourceName);
            if (resource == null) {
                throw new IllegalStateException("Le fichier " + resourceName + " n'a pas été trouvé dans resources.");
            }
            Path pdfPath = Paths.get(resource.toURI());

            DocumentParser parser = new ApacheTikaDocumentParser();
            Document document = FileSystemDocumentLoader.loadDocument(pdfPath, parser);

            DocumentSplitter splitter = DocumentSplitters.recursive(500, 50);
            List<TextSegment> segments = splitter.split(document);

            Response<List<Embedding>> response = embeddingModel.embedAll(segments);
            List<Embedding> embeddings = response.content();

            EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
            store.addAll(embeddings, segments);

            System.out.println("Ingestion terminée pour " + resourceName + " : "
                    + segments.size() + " segments enregistrés.");

            return store;
        } catch (URISyntaxException e) {
            throw new RuntimeException("Erreur lors du chargement du PDF " + resourceName, e);
        }
    }

    /**
     * Initialisation statique du RAG multi-PDF + Web (Tavily) + RAG conditionnel.
     */
    private static synchronized void ensureRagInitialized(String apiKeyGemini) {
        if (ragInitialized && retrievalAugmentor != null) {
            return;
        }

        // Modèle utilisé pour le routage conditionnel (comme dans Test4)
        ChatModel routingChatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKeyGemini)
                .modelName("gemini-2.5-flash")
                .temperature(0.0) // on veut une réponse la plus déterministe possible
                .logRequestsAndResponses(true)
                .build();

        // Modèle d'embedding partagé
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // --------- PHASE 1 : ingestion de 2 PDFs ---------
        EmbeddingStore<TextSegment> iaStore =
                ingestPdfAsEmbeddingStore(LlmClient.class, "langchain4j.pdf", embeddingModel);

        EmbeddingStore<TextSegment> autreStore =
                ingestPdfAsEmbeddingStore(LlmClient.class, "QCM_MAD-AI_COMPLET.pdf", embeddingModel);

        // --------- PHASE 2 : ContentRetrievers sur les PDFs ---------
        ContentRetriever iaRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(iaStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        ContentRetriever autreRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(autreStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        // --------- PHASE 3 : ContentRetriever Web (Tavily) ---------
        String tavilyKey = System.getenv("TAVILY_API_KEY");
        if (tavilyKey == null || tavilyKey.isBlank()) {
            throw new IllegalStateException("La variable d'environnement TAVILY_API_KEY n'est pas définie");
        }

        WebSearchEngine tavilyEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyKey)
                .build();

        ContentRetriever webRetriever =
                WebSearchContentRetriever.builder()
                        .webSearchEngine(tavilyEngine)
                        .build();

        // --------- PHASE 4 : PromptTemplate pour décider "RAG ou pas ?" ---------
        PromptTemplate routerTemplate = PromptTemplate.from(
                "Est-ce que la requête suivante nécessite d'utiliser le contexte des documents PDF " +
                        "ou d'effectuer une recherche Web (Tavily) ? " +
                        "Réponds uniquement par 'oui', 'non' ou 'peut-être'.\n" +
                        "Requête : {{query}}"
        );

        // --------- PHASE 5 : QueryRouter conditionnel ---------
        QueryRouter queryRouter = new QueryRouter() {
            @Override
            public List<ContentRetriever> route(Query query) {

                Prompt prompt = routerTemplate.apply(Map.of(
                        "query", query.text()
                ));

                String answer = routingChatModel.chat(prompt.text()).trim().toLowerCase();

                System.out.println("[Router conditionnel] Question utilisateur : " + query.text());
                System.out.println("[Router conditionnel] Réponse du LM pour le routage : " + answer);

                if (answer.startsWith("non")) {
                    // ❌ Pas de RAG : ni PDF, ni Web
                    return List.of();
                } else {
                    // ✅ "oui" ou "peut-être" → on utilise toutes les sources (2 PDFs + Web)
                    return List.of(iaRetriever, autreRetriever, webRetriever);
                }
            }
        };

        // --------- PHASE 6 : RetrievalAugmentor final ---------
        retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        ragInitialized = true;
    }

    // =========================
    //  INIT PARESSEUSE DU BEAN
    // =========================
    private void ensureInit() {
        if (assistant != null) {
            return;
        }

        this.apiKey = System.getenv("GEMINI_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("GEMINI_KEY manquante.");
        }

        // Modèle de chat utilisé pour répondre à l'utilisateur (appli Web)
        this.model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // Initialisation RAG (multi-PDF + Tavily + RAG conditionnel)
        ensureRagInitialized(this.apiKey);

        this.chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        if (this.systemRole != null && !this.systemRole.isBlank()) {
            this.chatMemory.add(SystemMessage.from(this.systemRole));
        }

        this.assistant = AiServices.builder(Assistant.class)
                .chatModel(this.model)
                .chatMemory(this.chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();
    }

    // =========================
    //  API UTILISÉE PAR LA WEBAPP
    // =========================

    public void setSystemRole(String role) {
        this.systemRole = role;
        ensureInit();
        this.chatMemory.clear();
        if (role != null && !role.isBlank()) {
            this.chatMemory.add(SystemMessage.from(role));
        }
    }

    public String ask(String prompt) {
        ensureInit();
        return assistant.chat(prompt);
    }
}
