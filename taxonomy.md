
# Intent Taxonomy — Uber_Support

Derived from HDBSCAN clustering (all-MiniLM-L6-v2 embeddings) over 41,660
customer root complaints, two passes (see decision_log.md items 6-9).
16 intents, locked.

---

### 1. `ride_cancellation_dispute`

Driver cancels (or effectively forces a cancellation) and the customer is
still charged a cancellation/fare fee; disputes over who's at fault.

> "Driver never arrived and cancelled himself but decided to charge me anyway."

### 2. `ride_fare_overcharge`

Fare charged is higher than quoted/upfront estimate, surge pricing disputes,
duplicate charges on a ride.

> "The fare charged was much higher than upfront fare for no reason."

### 3. `eats_delivery_issue`

Order late, cold, wrong address, wrong items — delivery execution problems
(not refund requests specifically).

> "My Dallas Ubereats driver delivered my order to the wrong building."

### 4. `eats_refund_request`

Food missing/wrong/never arrived, explicitly asking for money back.

> "I want my money back for the food I did not receive."

### 5. `payment_method_issue`

Card declined, can't add/remove/change payment method, unauthorized charges
on file.

> "Hello I'm trying to use the app, the app isn't accepting my credit card."

### 6. `promo_ridepass_issue`

Promo code not applying, Ride Pass signup/renewal/subscription problems.

> "I got an email for a promo code, and its not working."

### 7. `app_booking_technical`

Can't request/schedule a ride, verification code not sending, generic app
errors during booking.

> "I'M TRYNA GET TO SUBWAY & UR STUPID ASS APP WON'T LET ME REQUEST A RIDE."

### 8. `account_phone_verification`

Phone number already in use, can't receive verification code, login/signup
blocked.

> "I have a new phone number, how Can I get a verification code?"

### 9. `driver_document_onboarding`

Driver-side: document upload rejected, background check status, vehicle
registration/insurance issues.

> "What document do I need to upload for the Criminal Background Check."

### 10. `driver_safety_misconduct`

Harassment, physical altercation, dangerous driving, discrimination —
safety-critical. **Always escalate regardless of classifier confidence.**

> "This asshole legit kicked me and my friend out for singing in his car!"

### 11. `lost_item`

Left/lost a phone, wallet, or other item in the vehicle, needs help
contacting driver.

> "I left my phone in your uber last night. I tracked it..."

### 12. `rating_dispute`

Mistaken star rating given/received, disputes about rating impact on
account status. Originally also included tipping complaints; split into
`tip_issue` after a manual audit found the two mixed (decision_log.md #18).

> "Hello. So I mistakenly gave a driver 1 star rating. Is it possible to change that please?"

### 13. `uberpool_routing_issue`

UberPool-specific: bad routing, unwanted extra passengers, pool algorithm
complaints.

> "New pool algorithm is fully effed on west side of NYC."

### 14. `driver_vehicle_condition`

Vehicle cleanliness, smell, non-smoking violations — not safety/harassment.

> "My driver car smell like hot Garbage."

### 15. `general_complaint`

Vague frustration/venting with no specific actionable request; catch-all
for complaints that don't map to a concrete problem type.

> "Worst experience ever w/. How do they claim to innovate culturally?"

### 16. `tip_issue`

Tipping-related problems: can't leave a tip, tip option missing from app,
tip charged incorrectly, disputes about tip amount reaching the driver.
Split out of `rating_dispute` (decision_log.md #18).

> "we want to tip our driver but for some reason, the tip option isn't available"

---

## Notes for downstream use

- Intents 1, 2, 5 apply mostly to Rides; 3, 4 apply mostly to Eats;
  5, 6, 7, 8 are cross-cutting (apply to both products).
- Intent 10 (`driver_safety_misconduct`) bypasses normal confidence-based
  escalation — force-escalate on detection.
- Intent 15 (`general_complaint`) is the noisiest/largest catch-all;
  scattered errors across other intents (not one confusable pair) suggest
  this is inherent ambiguity, not a fixable data gap.
- Intent 16 (`tip_issue`) was added later, not part of the original
  clustering pass — added via manual audit, see decision_log.md #18.
