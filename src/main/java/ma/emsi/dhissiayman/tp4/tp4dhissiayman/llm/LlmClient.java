package ma.emsi.dhissiayman.tp4.tp4dhissiayman.llm;


import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.service.AiServices;
import jakarta.enterprise.context.Dependent;
import ma.emsi.dhissiayman.tp4.tp4dhissiayman.assistant.Assistant;


@Dependent
public class LlmClient implements java.io.Serializable {

    private static final long serialVersionUID = 1L;

    // Conserver uniquement l’état sérialisable
    private String systemRole;

    // Ces objets ne sont pas sérialisables -> transient
    private transient ChatMemory chatMemory;
    private transient Assistant assistant;

    // aussi généralement non sérialisable
    private transient ChatModel model;
    private transient String apiKey;

    // ---- init paresseuse ----
    private void ensureInit() {
        if (assistant != null) return;

        this.apiKey = System.getenv("GEMINI_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("GEMINI_KEY manquante.");
        }

        this.model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .build();

        this.chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        // Si un rôle système existait avant passivation, on le remet
        if (this.systemRole != null && !this.systemRole.isBlank()) {
            this.chatMemory.add(SystemMessage.from(this.systemRole));
        }

        this.assistant = AiServices.builder(Assistant.class)
                .chatModel(this.model)
                .chatMemory(this.chatMemory)
                .build();
    }

    public void setSystemRole(String role) {
        this.systemRole = role;
        // assistant peut être null après passivation -> réinit
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
