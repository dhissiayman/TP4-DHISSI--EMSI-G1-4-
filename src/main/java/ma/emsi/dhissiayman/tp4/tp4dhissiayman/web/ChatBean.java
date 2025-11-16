package ma.emsi.dhissiayman.tp4.tp4dhissiayman.web;



import jakarta.faces.application.FacesMessage;
import jakarta.faces.context.FacesContext;
import jakarta.faces.model.SelectItem;
import jakarta.faces.view.ViewScoped;
import jakarta.inject.Inject;
import jakarta.inject.Named;
import ma.emsi.dhissiayman.tp4.tp4dhissiayman.llm.LlmClient;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

@Named("bb")
@ViewScoped
public class ChatBean implements Serializable {

    // === dépendance vers LlmClient (métier) ===
    @Inject
    private LlmClient llm;

    // ==== UI : rôles système ====
    private List<SelectItem> listeRolesSysteme;
    private String roleSysteme;
    private boolean roleSystemeChangeable = true;

    // ==== états UI / conversation ====
    private boolean conversationDemarree = false;
    private String question;
    private String reponse;
    private String historique = "";

    @Inject private FacesContext faces;

    // === getters/setters utilisés par la page ===
    public String getRoleSysteme() { return roleSysteme; }
    public void setRoleSysteme(String r) { this.roleSysteme = r; }
    public boolean isRoleSystemeChangeable() { return roleSystemeChangeable; }
    public String getQuestion() { return question; }
    public void setQuestion(String q) { this.question = q; }
    public String getReponse() { return reponse; }
    public void setReponse(String r) { this.reponse = r; }
    public String getConversation() { return historique; }

    // === action "Envoyer" ===
    public void envoyer() {
        try {
            if (question == null || question.isBlank()) {
                faces.addMessage(null,
                        new FacesMessage(FacesMessage.SEVERITY_WARN,
                                "Question manquante","Veuillez saisir une question."));
                return;
            }

            // Au 1er envoi, pousser le rôle système vers LlmClient
            if (!conversationDemarree) {
                if (roleSysteme == null || roleSysteme.isBlank()) {
                    faces.addMessage(null,
                            new FacesMessage(FacesMessage.SEVERITY_WARN,
                                    "Rôle système manquant","Veuillez choisir un rôle de l'API."));
                    return;
                }
                llm.setSystemRole(roleSysteme);
                conversationDemarree = true;
                roleSystemeChangeable = false;
            }

            // Appel direct au LLM via AiServices
            this.reponse = llm.ask(this.question);

            // Mise à jour de l'historique affiché
            appendHistorique(this.question, this.reponse);

        } catch (Exception e) {
            faces.addMessage(null, new FacesMessage(
                    FacesMessage.SEVERITY_ERROR, "Erreur", e.getMessage()));
        }
    }

    // === action "Nouveau chat" ===
    public void nouveauChat() {
        this.roleSystemeChangeable = true;
        this.roleSysteme = null;
        this.question = null;
        this.reponse = null;
        this.historique = "";
        this.conversationDemarree = false;
        faces.addMessage(null, new FacesMessage(
                FacesMessage.SEVERITY_INFO, "Nouveau chat", "Session réinitialisée."));
    }

    // === helpers ===
    private void appendHistorique(String q, String r) {
        String bloc = "Vous: " + safe(q) + "\n" + "Modèle: " + safe(r) + "\n---\n";
        this.historique = (historique == null || historique.isBlank()) ? bloc : historique + bloc;
    }
    private String safe(String s) { return s == null ? "" : s; }

    // === liste déroulante des rôles (comme ton code) ===
    public List<SelectItem> getRolesSysteme() {
        if (this.listeRolesSysteme == null) {
            this.listeRolesSysteme = new ArrayList<>();
            String role = """
                You are a helpful and concise assistant.
                Answer clearly and help the user find information efficiently.
                """;
            this.listeRolesSysteme.add(new SelectItem(role, "Assistant"));

            role = """
                Tu es un traducteur FR-EN/EN-FR.
                Réponds uniquement par la traduction, sans explications.
                """;
            this.listeRolesSysteme.add(new SelectItem(role, "Traducteur Anglais-Français"));

            role = """
                Tu es un guide touristique local, chaleureux et précis.
                Donne des informations pratiques et concises.
                """;
            this.listeRolesSysteme.add(new SelectItem(role, "Guide Touristique"));
            // ... (tes autres rôles)
        }
        return this.listeRolesSysteme;
    }
}
